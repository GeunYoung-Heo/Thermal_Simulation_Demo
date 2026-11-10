"""Mesh reconstruction: TSDF fusion of the panoramic depth maps + marching cubes.

This is how RGB-D reconstructions (KinectFusion, Open3D, Matterport ...) are built, so the mesh
naturally carries the typical artefacts: lumpy surfaces from range noise, slightly doubled walls
from registration error, holes where nothing was observed (rack tops, behind objects).

After meshing, every vertex receives the label of its nearest labelled point (label transfer),
and the mesh is split into parts: one per detected object (with its ID) plus structure parts
(floor, walls, ceiling, clutter).  Parts are decimated for interactive display.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes

from .common import CLASS_INFO, STRUCTURE_COLORS
from .scanner import ScanData
from .segmentation import STRUCT_CODES, SegmentationResult


def tsdf_fusion(scan: ScanData, points, lo, hi, voxel=0.05, trunc=0.15, progress=None):
    origin = np.asarray(lo, float) - 2 * voxel
    dims = np.ceil((np.asarray(hi) - origin) / voxel).astype(int) + 3
    # only voxels near observed points are integrated (sparse band, like a hashed TSDF)
    key = np.floor((points - origin) / voxel).astype(np.int64)
    key = np.clip(key, 0, dims - 1)
    band = np.zeros(dims, bool)
    band[key[:, 0], key[:, 1], key[:, 2]] = True
    band = ndimage.binary_dilation(band, iterations=int(np.ceil(trunc / voxel)))
    vidx = np.flatnonzero(band)
    ijk = np.stack(np.unravel_index(vidx, dims), 1)
    centers = origin + (ijk + 0.5) * voxel
    tsdf = np.ones(len(vidx), np.float32)
    weight = np.zeros(len(vidx), np.float32)
    g = scan.geom
    M = len(scan.positions)
    for s in range(M):
        if progress:
            progress(s / M, f"TSDF 융합 {s + 1}/{M}")
        v = (centers - scan.positions[s]) @ scan.rotations[s]
        row, col, r = g.project(v)
        ok = (row >= 0) & (row < g.height)
        row = np.clip(row, 0, g.height - 1)
        D = scan.depth[s][row, col]
        sdf = D - r
        ok &= np.isfinite(D) & (sdf > -trunc)
        val = np.clip(sdf[ok] / trunc, -1, 1).astype(np.float32)
        w_new = (1.0 / np.maximum(D[ok], 1.0)).astype(np.float32)        # near views count more
        w_old = weight[ok]
        tsdf[ok] = (tsdf[ok] * w_old + val * w_new) / (w_old + w_new)
        weight[ok] = w_old + w_new
    vol = np.ones(dims, np.float32)
    wvol = np.zeros(dims, np.float32)
    vol.flat[vidx] = tsdf
    wvol.flat[vidx] = weight
    return vol, wvol, origin


def extract_mesh(vol, wvol, origin, voxel, min_weight=0.08):
    obs = wvol > min_weight
    mask = ndimage.binary_dilation(obs)            # skip far-away empty space (superset of what we keep)
    verts, faces, normals, _ = marching_cubes(vol, level=0.0, mask=mask, allow_degenerate=False)
    # Every marching-cubes vertex sits on a grid edge.  Keep a triangle only if both end voxels of all its
    # edges were actually observed; otherwise it is the artificial boundary between measured space and the
    # unobserved default value (it would appear as a back-facing duplicate behind floors / walls).
    a = np.clip(np.floor(verts).astype(np.int64), 0, np.array(vol.shape) - 1)
    b = np.clip(np.ceil(verts).astype(np.int64), 0, np.array(vol.shape) - 1)
    v_ok = obs[a[:, 0], a[:, 1], a[:, 2]] & obs[b[:, 0], b[:, 1], b[:, 2]]
    faces = faces[v_ok[faces].all(1)]
    used, inv = np.unique(faces.ravel(), return_inverse=True)
    verts = verts[used] * voxel + origin + 0.5 * voxel
    return verts.astype(np.float32), inv.reshape(-1, 3).astype(np.int32)


def part_key_of_points(seg: SegmentationResult):
    """Integer part code per point: 0..K-1 = detected instance,  K + j = structure code j, -1 = outlier."""
    K = len(seg.instances)
    code = np.where(seg.point_instance >= 0, seg.point_instance.astype(np.int32), -1)
    st = seg.point_struct.astype(np.int32)
    code = np.where((code < 0) & (st >= 0), K + st, code)
    return code


def part_names(seg: SegmentationResult):
    names = []
    for inst in seg.instances:
        names.append({"key": inst["id"], "kind": "object", "class": inst["class"], "role": inst["role"],
                      "color": CLASS_INFO[inst["class"]]["color"]})
    for s in STRUCT_CODES:
        names.append({"key": s, "kind": "structure", "class": s, "role": "structure", "color": STRUCTURE_COLORS[s]})
    return names


def _compact(verts, faces):
    used, inv = np.unique(faces.ravel(), return_inverse=True)
    return verts[used], inv.reshape(-1, 3).astype(np.int32)


def build_parts(verts, faces, points, colors, pcode, names, display_faces=260000, progress=None):
    good = pcode >= 0
    tree = cKDTree(points[good])
    plab = pcode[good]
    pcol = colors[good].astype(np.float32)
    _, nn = tree.query(verts, k=1, workers=-1)
    vlab = plab[nn]
    a, b, c = vlab[faces[:, 0]], vlab[faces[:, 1]], vlab[faces[:, 2]]
    flab = np.where(b == c, b, a)

    import fast_simplification
    parts = []
    total = len(faces)
    ids, counts = np.unique(flab, return_counts=True)
    # face budget: 35 % structure / 65 % objects, ~sqrt(size) within each group so small devices keep detail
    is_struct = np.array([names[l]["kind"] == "structure" for l in ids])
    w = np.sqrt(counts.astype(float))
    budget = np.zeros(len(ids))
    for grp, share in ((is_struct, 0.35), (~is_struct, 0.65)):
        if grp.any():
            budget[grp] = display_faces * share * w[grp] / w[grp].sum()
    for n, (l, cnt) in enumerate(zip(ids, counts)):
        if progress:
            progress(n / len(ids), f"메쉬 분할/간소화: {names[l]['key']}")
        f = faces[flab == l]
        v, f = _compact(verts, f)
        target = max(int(budget[n]), 1500)
        if len(f) > target * 1.1:
            red = 1.0 - target / len(f)
            v2, f2 = fast_simplification.simplify(v.astype(np.float32), f.astype(np.int32), target_reduction=float(red), agg=5)
            if len(f2) > 0:
                v, f = v2.astype(np.float32), f2.astype(np.int32)
        _, nn = tree.query(v, k=4, workers=-1)
        vc = pcol[nn].mean(1).astype(np.uint8)
        parts.append({**names[l], "verts": v, "faces": f, "vcolor": vc, "full_faces": int(cnt)})
    return parts, flab, total


def run_reconstruction(scan: ScanData, seg: SegmentationResult, cfg: dict, progress=None):
    voxel = cfg["voxel"]
    trunc = cfg["truncation"]
    room = seg.room
    lo = np.array([room["x0"], room["y0"], room["z0"]]) - 0.2
    hi = np.array([room["x1"], room["y1"], room["z1"]]) + 0.2
    pts = scan.points[seg.inlier] if seg.inlier is not None else scan.points

    def report(f, msg):
        if progress:
            progress(f, msg)

    vol, wvol, origin = tsdf_fusion(scan, pts, lo, hi, voxel, trunc, progress=lambda f, m: report(0.6 * f, m))
    report(0.6, "마칭 큐브로 표면 추출")
    verts, faces = extract_mesh(vol, wvol, origin, voxel)
    del vol, wvol
    pcode = part_key_of_points(seg)
    names = part_names(seg)
    report(0.7, "포인트 라벨을 메쉬로 전이")
    parts, flab, total = build_parts(verts, faces, scan.points, scan.colors, pcode, names, cfg.get("display_faces", 200000),
                                     progress=lambda f, m: report(0.75 + 0.25 * f, m))
    full = {"verts": verts, "faces": faces, "face_part": flab.astype(np.int16), "names": names}
    stats = {"full_vertices": int(len(verts)), "full_faces": int(total),
             "display_faces": int(sum(len(p["faces"]) for p in parts)), "tsdf_voxel": voxel}
    return parts, full, stats
