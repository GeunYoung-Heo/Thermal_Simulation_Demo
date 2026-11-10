"""Simulated panoramic RGB-D scanner (Matterport / terrestrial-laser-scanner style).

For every scan station a 360 deg equirectangular panorama is ray-cast against the ground-truth
boxes.  Realistic imperfections are added so the result *looks* like a real capture:
    * range noise that grows with distance
    * random drop-outs and failures at grazing angles
    * "flying pixels" (mixed returns) at depth discontinuities
    * small registration (pose) error per station -> slightly doubled surfaces
    * occlusion holes (tops of racks, behind objects) come for free from ray casting
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .scene import (MAT_CARDBOARD, MAT_CEILING, MAT_FLOOR, MAT_ROLLUP, MAT_WALL, MAT_WOOD, Scene)


@dataclass
class PanoGeometry:
    width: int
    height: int
    el_min: float      # radians
    el_max: float

    @property
    def daz(self):
        return 2 * np.pi / self.width

    @property
    def del_(self):
        return (self.el_max - self.el_min) / self.height

    def angles(self):
        az = -np.pi + (np.arange(self.width) + 0.5) * self.daz
        el = self.el_max - (np.arange(self.height) + 0.5) * self.del_
        return az, el

    def directions(self):
        az, el = self.angles()
        A, E = np.meshgrid(az, el)
        return np.stack([np.cos(E) * np.cos(A), np.cos(E) * np.sin(A), np.sin(E)], -1)

    def project(self, v):
        """world-direction vectors (N,3) in station frame -> (row, col, range); row/col may be out of image."""
        r = np.linalg.norm(v, axis=1)
        az = np.arctan2(v[:, 1], v[:, 0])
        el = np.arcsin(np.clip(v[:, 2] / np.maximum(r, 1e-9), -1, 1))
        col = np.floor((az + np.pi) / self.daz).astype(np.int64) % self.width
        row = np.floor((self.el_max - el) / self.del_).astype(np.int64)
        return row, col, r


@dataclass
class ScanData:
    geom: PanoGeometry
    positions: np.ndarray       # (M,3) station positions as *estimated* by registration
    rotations: np.ndarray       # (M,3,3) estimated station orientation
    rgb: np.ndarray             # (M,H,W,3) uint8 panorama colour images
    depth: np.ndarray           # (M,H,W) float32 measured range, NaN = no return
    gt_label: np.ndarray        # (M,H,W) int16 ground-truth instance (only used to fake SAM3)
    points: np.ndarray          # (N,3) float32 fused point cloud
    colors: np.ndarray          # (N,3) uint8
    point_station: np.ndarray   # (N,) int16
    point_gt: np.ndarray        # (N,) int16 ground-truth instance (evaluation only)


# --------------------------------------------------------------------------- ray casting

def _angular_bounds(lo, hi, c):
    """Conservative azimuth / elevation window of an AABB seen from c. Returns (az_ranges, el_min, el_max)."""
    dx0, dx1 = lo[0] - c[0], hi[0] - c[0]
    dy0, dy1 = lo[1] - c[1], hi[1] - c[1]
    dz0, dz1 = lo[2] - c[2], hi[2] - c[2]
    cx = np.array([dx0, dx1, dx1, dx0])
    cy = np.array([dy0, dy0, dy1, dy1])
    dmax = np.sqrt((cx ** 2 + cy ** 2).max())
    inside_xy = dx0 <= 0 <= dx1 and dy0 <= 0 <= dy1
    if inside_xy:
        if dz0 > 0:
            return None, np.arctan2(dz0, dmax), np.pi / 2
        if dz1 < 0:
            return None, -np.pi / 2, np.arctan2(dz1, dmax)
        return None, -np.pi / 2, np.pi / 2
    ex = min(max(0.0, dx0), dx1) if not (dx0 <= 0 <= dx1) else 0.0
    ey = min(max(0.0, dy0), dy1) if not (dy0 <= 0 <= dy1) else 0.0
    dmin = max(np.hypot(ex, ey), 1e-6)
    a = np.arctan2(cy, cx)
    rel = (a - a[0] + np.pi) % (2 * np.pi) - np.pi
    amin, amax = a[0] + rel.min(), a[0] + rel.max()
    el_max = np.arctan2(dz1, dmin if dz1 > 0 else dmax)
    el_min = np.arctan2(dz0, dmin if dz0 < 0 else dmax)
    return (amin, amax), el_min, el_max


def _col_ranges(g: PanoGeometry, az_rng):
    W = g.width
    if az_rng is None:
        return [(0, W)]
    amin, amax = az_rng
    j0 = int(np.floor((amin + np.pi) / g.daz)) - 1
    j1 = int(np.ceil((amax + np.pi) / g.daz)) + 1
    if j1 - j0 >= W:
        return [(0, W)]
    j0m, j1m = j0 % W, j1 % W
    if j0m < j1m:
        return [(j0m, j1m)]
    return [(j0m, W), (0, j1m)]


def cast_station(scene: Scene, c: np.ndarray, g: PanoGeometry, dirs: np.ndarray):
    H, W = g.height, g.width
    depth = np.full((H, W), np.inf, np.float32)
    hit_box = np.full((H, W), -1, np.int32)
    hit_ax = np.zeros((H, W), np.int8)
    with np.errstate(divide="ignore"):
        invd = np.where(np.abs(dirs) > 1e-12, 1.0 / dirs, 1e12)
    for k in range(scene.n_boxes):
        lo, hi = scene.lo[k], scene.hi[k]
        if np.all(lo <= c) and np.all(c <= hi):
            continue
        az_rng, el_min, el_max = _angular_bounds(lo, hi, c)
        i0 = max(0, int(np.floor((g.el_max - el_max) / g.del_)) - 1)
        i1 = min(H, int(np.ceil((g.el_max - el_min) / g.del_)) + 1)
        if i1 <= i0:
            continue
        for j0, j1 in _col_ranges(g, az_rng):
            ID = invd[i0:i1, j0:j1]
            t1 = (lo - c) * ID
            t2 = (hi - c) * ID
            tmin = np.minimum(t1, t2)
            tn = tmin.max(-1)
            tf = np.maximum(t1, t2).min(-1)
            cur = depth[i0:i1, j0:j1]
            upd = (tf >= tn) & (tn > 1e-4) & (tn < cur)
            if upd.any():
                cur[upd] = tn[upd]
                hit_box[i0:i1, j0:j1][upd] = k
                hit_ax[i0:i1, j0:j1][upd] = tmin.argmax(-1)[upd]
    return depth, hit_box, hit_ax


# --------------------------------------------------------------------------- appearance

def _hash_noise(p, scale):
    q = np.floor(p / scale)
    v = np.sin(q[:, 0] * 12.9898 + q[:, 1] * 78.233 + q[:, 2] * 37.719) * 43758.5453
    return v - np.floor(v)


def _texture(scene: Scene, p, box, ax):
    col = scene.color[box].copy()
    mat = scene.material[box]
    lf = 0.5 * (np.sin(p[:, 0] * 1.3) * np.sin(p[:, 1] * 1.7) + 1)       # low-frequency stain
    m = mat == MAT_FLOOR
    if m.any():
        col[m] *= (0.92 + 0.08 * lf[m] + 0.04 * _hash_noise(p[m], 0.05))[:, None]
        for ln in scene.floor_lines:
            a = 0 if ln["axis"] == "x" else 1
            o = 1 - a
            near = m & (np.abs(p[:, o] - ln["at"]) < 0.06) & (p[:, a] >= ln["from"]) & (p[:, a] <= ln["to"])
            col[near] = (0.93, 0.78, 0.12)
    m = mat == MAT_WALL
    if m.any():
        along = np.where(ax[m] == 1, p[m, 0], p[m, 1])
        seam = (along % 1.15) < 0.025
        c = col[m]
        c[seam] *= 0.80
        low = p[m, 2] < 1.0
        c[low] = (0.40, 0.50, 0.58)
        col[m] = c
    m = mat == MAT_CEILING
    if m.any():
        col[m] *= np.where((p[m, 1] % 2.0) < 0.06, 0.82, 1.0)[:, None]
    m = mat == MAT_ROLLUP
    if m.any():
        col[m] *= np.where((p[m, 2] % 0.18) < 0.025, 0.72, 1.0)[:, None]
    m = mat == MAT_CARDBOARD
    if m.any():
        col[m] *= (0.94 + 0.10 * _hash_noise(p[m], 0.12))[:, None]
    m = mat == MAT_WOOD
    if m.any():
        col[m] *= np.where((p[m, 0] % 0.14) < 0.035, 0.78, 1.0)[:, None]
    return col


# --------------------------------------------------------------------------- main entry

def _small_rotation(rng, sigma_rad):
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    ang = rng.normal(0, sigma_rad)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def scan(scene: Scene, progress=None) -> ScanData:
    sc = scene.cfg["scanner"]
    rng = np.random.default_rng(scene.cfg.get("seed", 0) + 101)
    g = PanoGeometry(sc["width"], sc["height"], np.radians(sc["elevation_range"][0]), np.radians(sc["elevation_range"][1]))
    dirs = g.directions()
    a, b = sc["range_noise"]
    M = len(scene.stations)
    H, W = g.height, g.width

    rgb_all = np.zeros((M, H, W, 3), np.uint8)
    depth_all = np.full((M, H, W), np.nan, np.float32)
    gt_all = np.full((M, H, W), -1, np.int16)
    pos_est = np.zeros((M, 3))
    rot_est = np.zeros((M, 3, 3))
    pts, cols, sts, gts = [], [], [], []

    for s, c in enumerate(scene.stations):
        if progress:
            progress(s / M, f"스캔 스테이션 {s + 1}/{M} 레이캐스팅")
        depth, box, ax = cast_station(scene, c, g, dirs)
        valid = np.isfinite(depth)
        p_true = c + dirs * np.where(valid, depth, 0)[..., None]

        # --- colour image (the RGB camera sees every pixel)
        bi = np.where(valid, box, 0).ravel()
        axf = ax.ravel()
        col = _texture(scene, p_true.reshape(-1, 3), bi, axf)
        cos_inc = np.abs(np.take_along_axis(dirs.reshape(-1, 3), axf[:, None].astype(np.int64), 1)[:, 0])
        shade = 0.55 + 0.45 * cos_inc
        col = col * shade[:, None] * (1 + rng.normal(0, 0.025, (len(col), 1)))
        img = np.clip(col * 255, 0, 255).astype(np.uint8).reshape(H, W, 3)
        img[~valid] = 0
        rgb_all[s] = img
        inst = np.where(valid, scene.inst[np.where(valid, box, 0)], -1)
        gt_all[s] = inst

        # --- range measurement with noise / failures
        r = depth.astype(np.float64)
        r_meas = r + rng.normal(0, 1, r.shape) * (a + b * np.where(valid, r, 0))
        ok = valid.copy()
        ok &= rng.random(r.shape) > sc["dropout"]
        graz = cos_inc.reshape(H, W) < sc["grazing_cutoff"]
        ok &= ~(graz & (rng.random(r.shape) < 0.7))
        ok &= np.where(valid, r, np.inf) < sc["max_range"]
        # flying pixels at depth discontinuities (both image directions)
        rr = np.where(valid, r, np.nan)
        for axis in (0, 1):
            nb = np.roll(rr, -1, axis=axis)
            jump = np.abs(nb - rr)
            edge = (jump > 0.25) & (jump > 0.05 * np.fmin(rr, nb))
            edge &= rng.random(r.shape) < sc["flying_pixel_prob"]
            near, far = np.fmin(rr, nb), np.fmax(rr, nb)
            r_meas = np.where(edge, near + rng.uniform(0.15, 0.85, r.shape) * (far - near), r_meas)
        depth_meas = np.where(ok, r_meas, np.nan).astype(np.float32)

        # --- registration error: the reconstruction believes a slightly wrong pose
        R = _small_rotation(rng, np.radians(sc["pose_noise"]["rotation_deg"]))
        t = rng.normal(0, sc["pose_noise"]["translation"], 3)
        pos_est[s] = c + t
        rot_est[s] = R
        depth_all[s] = depth_meas

        m = ok.ravel()
        local = dirs.reshape(-1, 3)[m] * depth_meas.ravel()[m, None]
        pts.append((local @ R.T + pos_est[s]).astype(np.float32))
        cols.append(img.reshape(-1, 3)[m])
        sts.append(np.full(m.sum(), s, np.int16))
        gts.append(inst.ravel()[m].astype(np.int16))

    pts = np.concatenate(pts)
    cols = np.concatenate(cols)
    sts = np.concatenate(sts)
    gts = np.concatenate(gts)

    # voxel down-sampling of the fused cloud (keeps one point per voxel)
    if progress:
        progress(0.98, "포인트클라우드 통합 및 다운샘플링")
    vs = sc.get("downsample", 0.02)
    key = np.floor((pts - pts.min(0)) / vs).astype(np.int64)
    key = (key[:, 0] * 100003 + key[:, 1]) * 100003 + key[:, 2]
    order = rng.permutation(len(pts))
    _, first = np.unique(key[order], return_index=True)
    keep = np.sort(order[first])

    return ScanData(
        geom=g, positions=pos_est, rotations=rot_est, rgb=rgb_all, depth=depth_all, gt_label=gt_all,
        points=pts[keep], colors=cols[keep], point_station=sts[keep], point_gt=gts[keep],
    )
