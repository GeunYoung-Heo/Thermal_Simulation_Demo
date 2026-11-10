"""Where should the thermal devices go, and at what temperature should they run?

Two nested problems:

1. Operating temperatures for a GIVEN placement  (exact, linear programming)
   Conduction is linear, so the field is   T(x) = T_base(x) + sum_i theta_i * G_i(x)
   where theta_i is the surface temperature of device i and G_i its unit response.
   Stage A  min s          s.t.  low + s' - s <= T(x) <= high - s' + s   for every zone cell x
                                theta_min <= theta_i <= theta_max
            (s <= 0 means the zone is inside the band; -s is the remaining safety margin)
   Stage B  min sum |Q_i|  s.t.  zone stays inside the band reached in stage A
            (Q_i = heat delivered by device i, also linear in theta  ->  minimum-energy operation)

2. Placement among candidate mounting positions  (greedy + swap search)
   To evaluate thousands of combinations quickly, each candidate is first modelled as a heat source
   occupying its future cells; its unit response H_c is computed once (one linear solve per candidate).
   Any subset is then scored by a small LP over the powers p_c (device surface temperature is the mean
   temperature of its cells, constrained to the device's allowed range).
   The best subset is finally re-evaluated exactly with Dirichlet devices (problem 1).
"""
from __future__ import annotations

import copy
import time

import numpy as np
from scipy.optimize import linprog

from .common import CLASS_INFO, WALL_FACES
from .thermal import (AIR, ThermalModel, SimResult, bc_temperatures, body_temperatures, build_grid,
                      zone_stats)
from .twin import device_box

WALLS = ["wall_south", "wall_north", "wall_west", "wall_east"]


# =============================================================================== candidates

def _overlap(alo, ahi, blo, bhi, pad=0.0):
    return np.all(alo - pad < bhi) and np.all(blo - pad < ahi)


def generate_candidates(twin: dict, cfg: dict, cls: str, spacing=2.0, walls=None, keep_devices=()):
    """Wall-mounted candidate positions for devices of class `cls`."""
    spec = cfg["thermal"]["device_types"][cls]
    tpl = twin["templates"][cls]
    L, T, H = tpl["L"], tpl["T"], tpl["H"]
    lo, hi = np.array(twin["domain"]["lo"]), np.array(twin["domain"]["hi"])
    walls = walls or WALLS
    obstacles = [(np.array(r["lo"]), np.array(r["hi"])) for r in twin["racks"]]
    obstacles += [(np.array(h["lo"]), np.array(h["hi"])) for h in twin["heat_sources"]]
    obstacles += [device_box(d) for d in keep_devices]
    cands = []
    for wall in walls:
        ax, side = WALL_FACES[wall]
        along = 0 if ax == 1 else 1
        yaw = 0 if ax == 1 else 90
        u_lo, u_hi = lo[along] + 0.4 + L / 2, hi[along] - 0.4 - L / 2
        n = max(1, int(np.floor((u_hi - u_lo) / spacing)) + 1)
        span = (n - 1) * spacing
        us = u_lo + (u_hi - u_lo - span) / 2 + np.arange(n) * spacing
        off = spec.get("wall_offset", 0.05) + T / 2
        for u in us:
            c = np.zeros(3)
            c[along] = u
            c[ax] = lo[ax] + off if side == 0 else hi[ax] - off
            c[2] = lo[2] + spec.get("mount_z", 0.15) + H / 2
            dev = {"x": c[0], "y": c[1], "z": c[2], "yaw": yaw, "L": L, "T": T, "H": H}
            blo, bhi = device_box(dev)
            bad = any(_overlap(blo, bhi, o_lo, o_hi, 0.15) for o_lo, o_hi in obstacles)
            for d in twin["doors"]:
                if d["wall"] != wall:
                    continue
                if blo[along] < d["u1"] + 0.2 and bhi[along] > d["u0"] - 0.2 and blo[2] < d["z1"] and bhi[2] > d["z0"]:
                    bad = True
            if not bad:
                cands.append({"cid": len(cands), "wall": wall, "u": round(float(u), 3),
                              "x": round(float(c[0]), 3), "y": round(float(c[1]), 3), "z": round(float(c[2]), 3),
                              "yaw": yaw, "L": L, "T": T, "H": H})
    return cands


# =============================================================================== problem 1

def _band_lp(Z0, ZG, low, high, bounds_theta, Q0=None, QM=None, s_cap=None):
    """LP over theta (+ slack s).  Z0: (m,) base zone temps, ZG: (m,n) unit responses.
    Returns theta, s.  If s_cap is given: minimise total |Q| subject to s <= s_cap."""
    m, n = ZG.shape
    # variables: theta (n), s (1) [, u (n) for |Q|]
    if s_cap is None:
        c = np.zeros(n + 1)
        c[-1] = 1.0
        A = np.vstack([np.hstack([-ZG, -np.ones((m, 1))]), np.hstack([ZG, -np.ones((m, 1))])])
        b = np.concatenate([Z0 - low, high - Z0])
        bnds = list(bounds_theta) + [(-(high - low), 1e3)]
        r = linprog(c, A_ub=A, b_ub=b, bounds=bnds, method="highs")
        if r.status != 0:
            return None, np.inf
        return r.x[:n], float(r.x[-1])
    c = np.concatenate([np.zeros(n + 1), np.ones(n)])
    A1 = np.hstack([-ZG, -np.ones((m, 1)), np.zeros((m, n))])
    A2 = np.hstack([ZG, -np.ones((m, 1)), np.zeros((m, n))])
    A3 = np.hstack([QM, np.zeros((n, 1)), -np.eye(n)])        #  Q - u <= 0
    A4 = np.hstack([-QM, np.zeros((n, 1)), -np.eye(n)])       # -Q - u <= 0
    A = np.vstack([A1, A2, A3, A4])
    b = np.concatenate([Z0 - low, high - Z0, -Q0, Q0])
    bnds = list(bounds_theta) + [(-(high - low), s_cap)] + [(0, None)] * n
    r = linprog(c, A_ub=A, b_ub=b, bounds=bnds, method="highs")
    if r.status != 0:
        return None, np.inf
    return r.x[:n], float(r.x[n])


def optimize_temperatures(twin, devices, bcs, cfg, h, k_air, k_goods, low, high, margin=0.5,
                          label="최적 운전온도", progress=None) -> SimResult:
    """Best operating temperature of every enabled device for the given placement."""
    t0 = time.time()
    dev_types = cfg["thermal"]["device_types"]
    grid = build_grid(twin, devices, bcs, h, k_air, k_goods)
    model = ThermalModel(grid, bcs)
    Tg = bc_temperatures(grid, bcs)
    Tb = body_temperatures(grid, devices, bcs)
    dev_pos = [i for i, k in enumerate(grid.body_kind) if k == "device"]
    by_id = {d["id"]: d for d in devices}
    Tb0 = Tb.copy()
    Tb0[dev_pos] = 0.0
    if progress:
        progress(0.1, "기준 온도장 계산 (장치 0 °C)")
    x0 = model.solve(Tg, Tb0)
    X = []
    for j, bi in enumerate(dev_pos):
        if progress:
            progress(0.1 + 0.6 * j / max(len(dev_pos), 1), f"장치 {grid.body_names[bi]} 단위 응답 계산")
        e = np.zeros(len(Tb))
        e[bi] = 1.0
        X.append(model.solve(np.zeros(len(Tg)), e))
    X = np.array(X).T if X else np.zeros((model.n, 0))
    zi = model.idx[grid.zone]
    zi = zi[zi >= 0]
    Z0, ZG = x0[zi], X[zi]
    # device heat  Q = Q0 + QM theta
    B = model.Bbody[:, dev_pos].tocsc()
    colsum = np.asarray(B.sum(0)).ravel()
    Q0 = -(B.T @ x0)
    QM = np.diag(colsum) - (B.T @ X)
    bounds = []
    for bi in dev_pos:
        spec = dev_types[by_id[grid.body_names[bi]]["class"]]
        bounds.append((spec["T_min"], spec["T_max"]))
    lo_b, hi_b = low + margin, high - margin
    if progress:
        progress(0.8, "LP: 관리구역 온도 범위 만족 + 에너지 최소화")
    if len(dev_pos) == 0:
        theta, s = np.zeros(0), max(float(lo_b - Z0.min()), float(Z0.max() - hi_b), -(hi_b - lo_b) / 2)
    else:
        theta, s = _band_lp(Z0, ZG, lo_b, hi_b, bounds)
        if theta is None:
            raise RuntimeError("LP failed")
        theta2, s2 = _band_lp(Z0, ZG, lo_b, hi_b, bounds, Q0, QM, s_cap=max(s, 0.0) + 1e-3)
        if theta2 is not None:
            theta, s = theta2, s2
    Tb_opt = Tb.copy()
    Tb_opt[dev_pos] = theta
    out_devices = copy.deepcopy(devices)
    for bi, th in zip(dev_pos, theta):
        by = {d["id"]: d for d in out_devices}
        by[grid.body_names[bi]]["T_set"] = round(float(th), 2)
    x = x0 + X @ theta
    T = model.field(x, Tb_opt)
    qb = model.body_heat(x, Tb_opt)
    qg = model.group_heat(x, Tg)
    meta = {
        "label": label,
        "target": {"low": low, "high": high, "margin": margin},
        "grid": {"h": h, "shape": list(grid.shape), "unknowns": model.n},
        "materials": {"air": k_air, "goods": k_goods},
        "body_heat_W": {n: round(float(q), 1) for n, q in zip(grid.body_names, qb)},
        "group_heat_W": {n: round(float(q), 1) for n, q in zip(grid.groups, qg)},
        "lp_violation": round(float(s), 3),
        "feasible": bool(s <= 1e-3),
        "device_power_W": round(float(sum(abs(qb[i]) for i in dev_pos)), 1),
        "solve_seconds": round(time.time() - t0, 2),
    }
    meta.update(zone_stats(T, grid.zone, low, high))
    return SimResult(T=T, grid=grid, devices=out_devices, bcs=bcs, meta=meta)


# =============================================================================== problem 2

def optimize_placement(twin, devices, bcs, cfg, cls, n_devices, candidates, h, k_air, k_goods,
                       low, high, margin=0.5, objective="margin", zone_sample=4000, seed=0, progress=None):
    """Choose `n_devices` of the candidate positions for devices of class `cls`, then optimise temperatures."""
    t0 = time.time()
    rng = np.random.default_rng(seed)
    spec = cfg["thermal"]["device_types"][cls]
    others = [d for d in devices if d["class"] != cls and d.get("enabled", True)]
    grid = build_grid(twin, others, bcs, h, k_air, k_goods)
    model = ThermalModel(grid, bcs)
    Tg = bc_temperatures(grid, bcs)
    Tb = body_temperatures(grid, others, bcs)
    if progress:
        progress(0.02, "기준 온도장 계산 (해당 장치 없음)")
    x0 = model.solve(Tg, Tb)

    # unit responses of every candidate (1 kW spread over its future cells)
    valid, H, cells = [], [], []
    for n, c in enumerate(candidates):
        if progress:
            progress(0.05 + 0.45 * n / max(len(candidates), 1), f"후보 위치 {n + 1}/{len(candidates)} 응답 계산")
        blo, bhi = device_box(c)
        sl = grid.index_range(blo, bhi)
        if np.any(grid.cell_type[sl] != AIR):
            continue
        ids = model.idx[sl].ravel()
        src = np.zeros(model.n)
        src[ids] = 1000.0 / len(ids)
        H.append(model.solve_rhs(src))
        cells.append(ids)
        valid.append(c)
    if len(valid) < n_devices:
        raise ValueError(f"사용 가능한 후보 위치({len(valid)})가 장치 수({n_devices})보다 적습니다.")
    H = np.array(H).T                                            # (n_unknowns, M)  [K per kW]
    zi = model.idx[grid.zone]
    zi = zi[zi >= 0]
    if len(zi) > zone_sample:
        zi = rng.choice(zi, zone_sample, replace=False)
    Z0, ZH = x0[zi], H[zi]
    D0 = np.array([x0[c].mean() for c in cells])                 # device temperature proxies
    DH = np.array([H[c].mean(0) for c in cells])                 # (M, M)
    lo_b, hi_b = low + margin, high - margin
    heating = spec["T_max"] > 30
    pbound = (0.0, 60.0) if heating else (-60.0, 0.0)

    def score(S):
        S = list(S)
        k = len(S)
        A_band = np.vstack([np.hstack([-ZH[:, S], -np.ones((len(zi), 1))]), np.hstack([ZH[:, S], -np.ones((len(zi), 1))])])
        b_band = np.concatenate([Z0 - lo_b, hi_b - Z0])
        Dsub = DH[np.ix_(S, S)]
        A_dev = np.vstack([np.hstack([Dsub, np.zeros((k, 1))]), np.hstack([-Dsub, np.zeros((k, 1))])])
        b_dev = np.concatenate([spec["T_max"] - D0[S], D0[S] - spec["T_min"]])
        A = np.vstack([A_band, A_dev])
        b = np.concatenate([b_band, b_dev])
        bnds = [pbound] * k + [(-(hi_b - lo_b), 1e3)]
        c = np.zeros(k + 1)
        c[-1] = 1.0
        r = linprog(c, A_ub=A, b_ub=b, bounds=bnds, method="highs")
        if r.status != 0:
            return (np.inf, np.inf)
        s = float(r.x[-1])
        if objective == "energy":
            c2 = np.concatenate([np.abs(np.ones(k)), [0.0]])
            bnds2 = [pbound] * k + [(-(hi_b - lo_b), max(s, 0.0) + 1e-3)]
            r2 = linprog(c2 if heating else -c2, A_ub=A, b_ub=b, bounds=bnds2, method="highs")
            power = float(np.abs(r2.x[:k]).sum()) if r2.status == 0 else np.inf
            return (max(s, 0.0), power)
        return (s, float(np.abs(r.x[:k]).sum()))

    def better(a, b):
        return (a[0] < b[0] - 1e-4) or (abs(a[0] - b[0]) <= 1e-4 and a[1] < b[1] - 1e-6)

    M = len(valid)
    S, history = [], []
    for step in range(n_devices):
        best, best_c = (np.inf, np.inf), None
        for c in range(M):
            if c in S:
                continue
            sc = score(S + [c])
            if better(sc, best):
                best, best_c = sc, c
        S.append(best_c)
        history.append({"step": f"추가 {step + 1}", "set": [valid[i]["cid"] for i in S], "violation": round(best[0], 3), "power_kW": round(best[1], 2)})
        if progress:
            progress(0.5 + 0.3 * (step + 1) / n_devices, f"탐욕 탐색 {step + 1}/{n_devices}: 위반량 {best[0]:.2f} °C")
    cur = score(S)
    improved, it = True, 0
    while improved and it < 4:
        improved, it = False, it + 1
        for i in range(len(S)):
            for c in range(M):
                if c in S:
                    continue
                T_ = S.copy()
                T_[i] = c
                sc = score(T_)
                if better(sc, cur):
                    S, cur, improved = T_, sc, True
        history.append({"step": f"교환 {it}", "set": [valid[i]["cid"] for i in S], "violation": round(cur[0], 3), "power_kW": round(cur[1], 2)})
        if progress:
            progress(0.8 + 0.05 * it, f"교환 탐색 {it}: 위반량 {cur[0]:.2f} °C")

    # exact evaluation with Dirichlet devices
    prefix = CLASS_INFO[cls]["prefix"]
    new_devs = [d for d in copy.deepcopy(devices) if d["class"] != cls]
    for n, i in enumerate(sorted(S, key=lambda i: (valid[i]["wall"], valid[i]["u"]))):
        c = valid[i]
        new_devs.append({"id": f"{prefix}-{n + 1:02d}", "class": cls, "enabled": True,
                         "x": c["x"], "y": c["y"], "z": c["z"], "yaw": c["yaw"], "L": c["L"], "T": c["T"], "H": c["H"],
                         "T_set": spec["T_default"], "mesh_ref": twin["templates"][cls].get("mesh_ref"),
                         "wall": c["wall"], "candidate": c["cid"]})
    if progress:
        progress(0.9, "선택된 배치에서 정확한 운전온도 최적화")
    res = optimize_temperatures(twin, new_devs, bcs, cfg, h, k_air, k_goods, low, high, margin, label="최적 배치 + 운전온도")
    res.meta["placement"] = {
        "class": cls, "n_devices": n_devices, "objective": objective, "n_candidates": M,
        "chosen": [valid[i]["cid"] for i in S], "history": history,
        "search_seconds": round(time.time() - t0, 2),
    }
    return res
