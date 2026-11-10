"""Steady-state heat conduction on a voxel grid (finite volume).

    div( k grad T ) = 0      inside the room (air + goods), conduction only

Cell types
    0 air       unknown temperature, k = k_air (effective conductivity incl. mixing)
    1 goods     unknown temperature, k = k_goods (inside racks)
    2 device    fixed temperature body   (radiator / air conditioner with an ID)  -> Dirichlet
    3 heat src  fixed temperature body   (charger, cabinet: no ID, boundary condition)
    4 excluded  insulated solid (heat source switched to 'adiabatic')

Domain faces (walls, floor, ceiling, door patches) carry boundary-condition groups:
    fixed       T_surface = T                    (U = inf)
    convective  q = U (T_ext - T_surface)        (Robin; U = overall heat-transfer coeff. [W/m2K])
    adiabatic   q = 0                            (U = 0)

The linear system A x = b is symmetric positive definite and solved with algebraic multigrid
(pyamg) + CG.  Because the problem is linear, the temperature field is a linear function of all
boundary temperatures - this is exploited by the optimizer.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from .common import WALL_FACES
from .twin import device_box

AIR, GOODS, DEVICE, HEAT_SOURCE, EXCLUDED = 0, 1, 2, 3, 4
FACE_ORDER = ["wall_west", "wall_east", "wall_south", "wall_north", "floor", "ceiling"]


@dataclass
class Grid:
    origin: np.ndarray
    h: float
    shape: tuple
    cell_type: np.ndarray            # int8 (nx,ny,nz)
    body: np.ndarray                 # int16 (nx,ny,nz) index into body_names, -1 none
    body_names: list
    body_kind: list                  # 'device' | 'heat_source'
    zone: np.ndarray                 # bool (nx,ny,nz)
    k: np.ndarray                    # float32 (nx,ny,nz)
    groups: list                     # BC group names
    face_group: dict                 # face name -> int16 2D array (group index)

    def centers(self, axis):
        return self.origin[axis] + (np.arange(self.shape[axis]) + 0.5) * self.h

    def index_range(self, lo, hi):
        """cell index ranges covering an AABB (at least one cell per axis)."""
        out = []
        for a in range(3):
            n = self.shape[a]
            i0 = int(np.ceil((lo[a] - self.origin[a]) / self.h - 0.5))
            i1 = int(np.floor((hi[a] - self.origin[a]) / self.h - 0.5))
            if i1 < i0:
                mid = 0.5 * (lo[a] + hi[a])
                i0 = i1 = int(np.floor((mid - self.origin[a]) / self.h))
            out.append(slice(int(np.clip(i0, 0, n - 1)), int(np.clip(i1, 0, n - 1)) + 1))
        return tuple(out)


def build_grid(twin: dict, devices: list, bcs: list, h: float, k_air: float, k_goods: float) -> Grid:
    lo = np.array(twin["domain"]["lo"], float)
    hi = np.array(twin["domain"]["hi"], float)
    shape = tuple(int(max(2, round(v))) for v in (hi - lo) / h)
    g = Grid(origin=lo, h=h, shape=shape,
             cell_type=np.zeros(shape, np.int8), body=np.full(shape, -1, np.int16),
             body_names=[], body_kind=[], zone=np.zeros(shape, bool),
             k=np.full(shape, k_air, np.float32), groups=[], face_group={})
    bc_by_name = {b["name"]: b for b in bcs}

    for r in twin["racks"]:
        sl = g.index_range(r["lo"], r["hi"])
        g.cell_type[sl] = GOODS
        g.k[sl] = k_goods
        if r.get("in_zone", True):
            g.zone[sl] = True

    for hs in twin["heat_sources"]:
        sl = g.index_range(hs["lo"], hs["hi"])
        b = bc_by_name.get(hs["name"], {"type": "fixed"})
        if b["type"] == "adiabatic":
            g.cell_type[sl] = EXCLUDED
        else:
            g.cell_type[sl] = HEAT_SOURCE
            g.body[sl] = len(g.body_names)
            g.body_names.append(hs["name"])
            g.body_kind.append("heat_source")

    for d in devices:
        if not d.get("enabled", True):
            continue
        dlo, dhi = device_box(d)
        sl = g.index_range(dlo, dhi)
        g.cell_type[sl] = DEVICE
        g.body[sl] = len(g.body_names)
        g.body_names.append(d["id"])
        g.body_kind.append("device")

    g.zone &= (g.cell_type == GOODS) | (g.cell_type == AIR)

    # boundary groups on the six domain faces; doors override wall patches
    g.groups = list(FACE_ORDER)
    for face in FACE_ORDER:
        ax, side = WALL_FACES[face]
        other = [a for a in range(3) if a != ax]
        g.face_group[face] = np.full((shape[other[0]], shape[other[1]]), g.groups.index(face), np.int16)
    for d in twin["doors"]:
        ax, side = WALL_FACES[d["wall"]]
        other = [a for a in range(3) if a != ax]          # (u axis, z axis)
        cu = g.centers(other[0])
        cz = g.centers(other[1])
        mu = (cu >= d["u0"]) & (cu <= d["u1"])
        mz = (cz >= d["z0"]) & (cz <= d["z1"])
        if not mu.any() or not mz.any():
            continue
        gi = len(g.groups)
        g.groups.append(d["name"])
        g.face_group[d["wall"]][np.ix_(mu, mz)] = gi
    return g


def _face_cells(arr, face):
    ax, side = WALL_FACES[face]
    idx = [slice(None)] * 3
    idx[ax] = 0 if side == 0 else -1
    return arr[tuple(idx)]


class ThermalModel:
    """Assembled linear system for a fixed geometry and fixed U-values."""

    def __init__(self, grid: Grid, bcs: list):
        self.grid = g = grid
        h = g.h
        unk = (g.cell_type == AIR) | (g.cell_type == GOODS)
        self.unknown = unk
        n = int(unk.sum())
        self.n = n
        idx = np.full(g.shape, -1, np.int64)
        idx[unk] = np.arange(n)
        self.idx = idx
        k = g.k.astype(np.float64)
        body = g.body.astype(np.int64)
        bc_by_name = {b["name"]: b for b in bcs}

        rows, cols, vals, drows, dvals = [], [], [], [], []
        brow, bcol, bval = [], [], []
        for ax in range(3):
            a = [slice(None)] * 3
            b = [slice(None)] * 3
            a[ax] = slice(0, -1)
            b[ax] = slice(1, None)
            a, b = tuple(a), tuple(b)
            ia, ib = idx[a], idx[b]
            ka, kb = k[a], k[b]
            ba, bb = body[a], body[b]
            m = (ia >= 0) & (ib >= 0)
            G = h * 2 * ka[m] * kb[m] / (ka[m] + kb[m])
            rows += [ia[m], ib[m]]
            cols += [ib[m], ia[m]]
            vals += [-G, -G]
            drows += [ia[m], ib[m]]
            dvals += [G, G]
            for iu, ku, bo in ((ia, ka, bb), (ib, kb, ba)):
                m = (iu >= 0) & (bo >= 0)
                G = 2 * h * ku[m]
                drows.append(iu[m])
                dvals.append(G)
                brow.append(iu[m])
                bcol.append(bo[m])
                bval.append(G)

        crow, ccol, cval = [], [], []
        U_of_group = np.zeros(len(g.groups))
        for gi, name in enumerate(g.groups):
            bc = bc_by_name.get(name)
            if bc is None:
                U_of_group[gi] = 0.0
            elif bc["type"] == "fixed":
                U_of_group[gi] = np.inf
            elif bc["type"] == "adiabatic":
                U_of_group[gi] = 0.0
            else:
                U_of_group[gi] = float(bc["U"])
        self.U_of_group = U_of_group
        for face in FACE_ORDER:
            fi = _face_cells(idx, face)
            fk = _face_cells(k, face)
            fg = g.face_group[face].astype(np.int64)
            m = fi >= 0
            U = U_of_group[fg[m]]
            kk = fk[m]
            with np.errstate(divide="ignore", invalid="ignore"):
                G = np.where(np.isinf(U), 2 * h * kk, np.where(U > 0, h * h / (1.0 / np.maximum(U, 1e-12) + h / (2 * kk)), 0.0))
            keep = G > 0
            drows.append(fi[m][keep])
            dvals.append(G[keep])
            crow.append(fi[m][keep])
            ccol.append(fg[m][keep])
            cval.append(G[keep])

        rows = np.concatenate(rows) if rows else np.zeros(0, np.int64)
        cols = np.concatenate(cols) if cols else np.zeros(0, np.int64)
        vals = np.concatenate(vals) if vals else np.zeros(0)
        diag = np.bincount(np.concatenate(drows), weights=np.concatenate(dvals), minlength=n)
        diag += 1e-9 * h * k[unk]          # tiny regularisation for isolated pockets
        A = sp.csr_matrix((vals, (rows, cols)), shape=(n, n)) + sp.diags(diag)
        self.A = A.tocsr()
        nb = len(g.body_names)
        self.Bbody = sp.csr_matrix((np.concatenate(bval) if bval else [], (np.concatenate(brow) if brow else [], np.concatenate(bcol) if bcol else [])), shape=(n, nb))
        self.Bbc = sp.csr_matrix((np.concatenate(cval), (np.concatenate(crow), np.concatenate(ccol))), shape=(n, len(g.groups)))
        self._ml = None

    # ------------------------------------------------------------------ solving
    @property
    def ml(self):
        if self._ml is None:
            import pyamg
            self._ml = pyamg.smoothed_aggregation_solver(self.A, symmetry="hermitian", max_coarse=400)
        return self._ml

    def solve_rhs(self, b, tol=1e-9):
        if not np.any(b):
            return np.zeros(self.n)
        res = []
        x = self.ml.solve(b, tol=tol, accel="cg", maxiter=500, residuals=res)
        return x

    def rhs(self, T_groups, T_bodies, source=None):
        b = self.Bbc @ np.asarray(T_groups, float)
        if self.Bbody.shape[1]:
            b = b + self.Bbody @ np.asarray(T_bodies, float)
        if source is not None:
            b = b + source
        return b

    def solve(self, T_groups, T_bodies, source=None):
        return self.solve_rhs(self.rhs(T_groups, T_bodies, source))

    # ------------------------------------------------------------------ post-processing
    def field(self, x, T_bodies):
        g = self.grid
        T = np.full(g.shape, np.nan, np.float32)
        T[self.unknown] = x
        if len(T_bodies):
            m = g.body >= 0
            T[m] = np.asarray(T_bodies, np.float32)[g.body[m]]
        return T

    def body_heat(self, x, T_bodies):
        """Heat delivered by every body into the room [W] (positive = heating)."""
        if not self.Bbody.shape[1]:
            return np.zeros(0)
        colsum = np.asarray(self.Bbody.sum(0)).ravel()
        return colsum * np.asarray(T_bodies, float) - self.Bbody.T @ x

    def group_heat(self, x, T_groups):
        """Heat entering the room through every BC group [W] (negative = heat loss)."""
        colsum = np.asarray(self.Bbc.sum(0)).ravel()
        return colsum * np.asarray(T_groups, float) - self.Bbc.T @ x


# ---------------------------------------------------------------------- convenience

def bc_temperatures(grid: Grid, bcs: list):
    by = {b["name"]: b for b in bcs}
    Tg = np.array([float(by[n]["T"]) if n in by and by[n].get("T") is not None else 0.0 for n in grid.groups])
    return Tg


def body_temperatures(grid: Grid, devices: list, bcs: list):
    by_bc = {b["name"]: b for b in bcs}
    by_dev = {d["id"]: d for d in devices}
    Tb = []
    for name, kind in zip(grid.body_names, grid.body_kind):
        if kind == "device":
            Tb.append(float(by_dev[name]["T_set"]))
        else:
            Tb.append(float(by_bc[name]["T"]))
    return np.array(Tb)


@dataclass
class SimResult:
    T: np.ndarray                 # (nx,ny,nz) float32, NaN in excluded cells
    grid: Grid
    devices: list
    bcs: list
    meta: dict = field(default_factory=dict)


def zone_stats(T, zone, low, high):
    z = T[zone]
    z = z[np.isfinite(z)]
    if len(z) == 0:
        return {}
    return {
        "zone_min": round(float(z.min()), 2), "zone_max": round(float(z.max()), 2),
        "zone_mean": round(float(z.mean()), 2),
        "zone_in_band": round(float(((z >= low) & (z <= high)).mean()), 4),
        "zone_below": round(float((z < low).mean()), 4), "zone_above": round(float((z > high).mean()), 4),
        "zone_cells": int(len(z)),
    }


def simulate(twin: dict, devices: list, bcs: list, h: float, k_air: float, k_goods: float,
             target=(15.0, 25.0), label: str = "simulation") -> SimResult:
    import time
    t0 = time.time()
    grid = build_grid(twin, devices, bcs, h, k_air, k_goods)
    model = ThermalModel(grid, bcs)
    Tg = bc_temperatures(grid, bcs)
    Tb = body_temperatures(grid, devices, bcs)
    x = model.solve(Tg, Tb)
    T = model.field(x, Tb)
    qb = model.body_heat(x, Tb)
    qg = model.group_heat(x, Tg)
    meta = {
        "label": label,
        "target": {"low": target[0], "high": target[1]},
        "grid": {"h": h, "shape": list(grid.shape), "unknowns": model.n},
        "materials": {"air": k_air, "goods": k_goods},
        "body_heat_W": {n: round(float(q), 1) for n, q in zip(grid.body_names, qb)},
        "group_heat_W": {n: round(float(q), 1) for n, q in zip(grid.groups, qg)},
        "solve_seconds": round(time.time() - t0, 2),
    }
    meta.update(zone_stats(T, grid.zone, *target))
    return SimResult(T=T, grid=grid, devices=devices, bcs=bcs, meta=meta)
