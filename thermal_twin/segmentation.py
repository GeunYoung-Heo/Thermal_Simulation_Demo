"""Simulated SAM3 detection + re-projection onto the point cloud.

Real pipeline being imitated:
    1. SAM3 is run on every panorama with text prompts ("radiator", "pallet rack" ...)
       -> per-image instance masks + class + confidence
    2. Each 3D point is projected into every panorama in which it is visible
       (depth-consistency test) and collects the class of the mask it lands in   (multi-view voting)
    3. Points of the same class are clustered in 3D -> object instances -> IDs (RAD-01, ...)
    4. Points that received no label are "structure": dominant planes give floor / ceiling / walls.

Step 1 is faked from the ground-truth label image, with realistic errors:
missed detections (small / far objects), dilated / eroded masks, noisy boundaries, varying scores.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from .common import CLASS_INFO, CLASS_NAMES, STRUCTURE_PARTS
from .scanner import ScanData

STRUCT_CODES = ["floor", "ceiling", "wall_south", "wall_north", "wall_west", "wall_east", "clutter"]
FLOOR_STANDING = {"pallet_rack", "battery_charger", "electrical_cabinet"}


@dataclass
class SegmentationResult:
    det_map: np.ndarray            # (M,H,W) int16 index into detections, -1 = none
    detections: list               # [{station, class, score, bbox:[r0,c0,r1,c1], pixels}]
    point_class: np.ndarray        # (N,) int8  voted class id, -1 = background
    point_instance: np.ndarray     # (N,) int16 index into instances, -1 = none
    point_struct: np.ndarray       # (N,) int8  index into STRUCT_CODES for background points, -1 otherwise
    instances: list                # [{id, class, role, n_points, confidence, centroid, lo, hi, gt_name, iou}]
    room: dict                     # detected room planes {x0,x1,y0,y1,z0,z1}
    stats: dict = field(default_factory=dict)
    inlier: np.ndarray = None      # (N,) bool, False = removed as outlier (flying pixel)


# --------------------------------------------------------------------------- 1. fake SAM3

def simulate_sam3(scan: ScanData, gt_instances: list, cfg: dict, seed: int = 0):
    rng = np.random.default_rng(seed + 202)
    M, H, W = scan.gt_label.shape
    det_map = np.full((M, H, W), -1, np.int16)
    score_map = np.zeros((H, W), np.float32)
    dets = []
    st3 = ndimage.generate_binary_structure(2, 1)
    for s in range(M):
        gt = scan.gt_label[s]
        score_map[:] = 0
        ids, counts = np.unique(gt[gt >= 0], return_counts=True)
        for inst, npx in zip(ids, counts):
            if npx < cfg["min_pixels"]:
                continue
            p_miss = cfg["miss_prob_small"] if npx < 150 else cfg["miss_prob"]
            if rng.random() < p_miss:
                continue
            rows, cols = np.nonzero(gt == inst)
            # handle panorama wrap-around: if the mask touches both image borders, shift it
            shift = 0
            if cols.min() == 0 and cols.max() == W - 1:
                shift = W // 2
            cs = (cols + shift) % W
            r0, r1 = max(rows.min() - 4, 0), min(rows.max() + 5, H)
            c0, c1 = max(cs.min() - 4, 0), min(cs.max() + 5, W)
            crop = np.zeros((r1 - r0, c1 - c0), bool)
            crop[rows - r0, cs - c0] = True
            op = rng.random()
            if op < 0.35:
                crop = ndimage.binary_dilation(crop, st3, iterations=int(rng.integers(1, 3)))
            elif op < 0.6:
                er = ndimage.binary_erosion(crop, st3)
                if er.sum() > 0.5 * crop.sum():
                    crop = er
            inner = crop & ~ndimage.binary_erosion(crop, st3)
            outer = ndimage.binary_dilation(crop, st3) & ~crop
            crop[inner & (rng.random(crop.shape) < cfg["boundary_flip"])] = False
            crop[outer & (rng.random(crop.shape) < cfg["boundary_flip"])] = True
            score = float(np.clip(0.55 + 0.42 * (1 - np.exp(-npx / 900)) + rng.normal(0, 0.04), 0.3, 0.99))
            if score < cfg["score_threshold"]:
                continue
            k = len(dets)
            rr, cc = np.nonzero(crop)
            rr = rr + r0
            cc = (cc + c0 - shift) % W
            better = score > score_map[rr, cc]
            det_map[s, rr[better], cc[better]] = k
            score_map[rr[better], cc[better]] = score
            cls = gt_instances[inst]["class"]
            dets.append({
                "station": s, "class": cls, "score": round(score, 3), "pixels": int(len(rr)),
                "bbox": [int(r0), int((c0 - shift) % W), int(r1), int((c1 - shift) % W)],
                "wrap": bool(shift),
            })
    return det_map, dets


# --------------------------------------------------------------------------- 2. multi-view voting

def vote_point_classes(scan: ScanData, det_map, dets, vote_ratio=0.4):
    g = scan.geom
    N = len(scan.points)
    det_cls = np.array([CLASS_NAMES.index(d["class"]) for d in dets] + [-1], np.int16)   # -1 index -> none
    votes = np.zeros((N, len(CLASS_NAMES)), np.float32)
    seen = np.zeros(N, np.float32)
    nseen = np.zeros(N, np.uint8)
    idx_all = np.arange(N)
    P = scan.points.astype(np.float64)
    for s in range(len(scan.positions)):
        v = (P - scan.positions[s]) @ scan.rotations[s]
        row, col, r = g.project(v)
        inimg = (row >= 0) & (row < g.height)
        ii = idx_all[inimg]
        row, col, r = row[inimg], col[inimg], r[inimg]
        D = scan.depth[s][row, col]
        vis = np.isfinite(D) & (np.abs(D - r) < 0.03 + 0.012 * r)
        ii, row, col, r = ii[vis], row[vis], col[vis], r[vis]
        # a far view sees the object with coarse pixels -> mask boundaries are less reliable
        w = np.clip(4.0 / r, 0.05, 1.0).astype(np.float32)
        seen[ii] += w
        nseen[ii] += 1
        cls = det_cls[det_map[s][row, col]]
        m = cls >= 0
        votes[ii[m], cls[m]] += w[m]
    best = votes.argmax(1)
    bv = votes.max(1)
    lab = np.where((bv > 0) & (bv >= vote_ratio * np.maximum(seen, 1e-6)), best, -1).astype(np.int8)
    conf = np.where(lab >= 0, bv / np.maximum(seen, 1e-6), 0).astype(np.float32)
    return lab, conf, nseen


# --------------------------------------------------------------------------- 0. outlier removal

def remove_outliers(points, k=8, factor=2.5, floor=0.06):
    """Statistical outlier removal: drop points whose mean k-NN distance is unusually large.
    Mostly removes 'flying pixels' (mixed returns floating between surfaces)."""
    from scipy.spatial import cKDTree
    tree = cKDTree(points)
    d, _ = tree.query(points, k=k + 1, workers=-1)
    dm = d[:, 1:].mean(1)
    thr = max(floor, float(np.median(dm)) * factor)
    return dm < thr


# --------------------------------------------------------------------------- 3. 3D instances

def cluster_instances(points, point_class, conf, vox=0.1, min_points=150, min_conf=0.6):
    inst_of_point = np.full(len(points), -1, np.int16)
    raw = []
    for cid, cls in enumerate(CLASS_NAMES):
        sel = np.flatnonzero(point_class == cid)
        if len(sel) < min_points:
            continue
        P = points[sel]
        o = P.min(0)
        key = np.floor((P - o) / vox).astype(np.int64)
        dims = key.max(0) + 1
        occ = np.zeros(dims, bool)
        occ[key[:, 0], key[:, 1], key[:, 2]] = True
        occd = ndimage.binary_dilation(occ, iterations=1)
        lab, n = ndimage.label(occd, structure=np.ones((3, 3, 3)))
        comp = lab[key[:, 0], key[:, 1], key[:, 2]]
        for c in range(1, n + 1):
            m = comp == c
            if m.sum() < min_points or conf[sel[m]].mean() < min_conf:
                continue          # tiny or low-confidence cluster -> treated as a false positive
            raw.append((cls, sel[m]))
    # geometric prior: racks / chargers / cabinets stand on the floor -> floating clusters are false positives
    floor_z = float(np.percentile(points[:, 2], 1.0))
    raw = [(c, i) for c, i in raw
           if c not in FLOOR_STANDING or np.percentile(points[i, 2], 1.0) < floor_z + 0.4]
    # order: by class, then x, then y  -> stable, human-friendly IDs
    raw.sort(key=lambda t: (CLASS_NAMES.index(t[0]), round(float(np.median(points[t[1], 1]))), float(np.median(points[t[1], 0]))))
    instances = []
    counters = {}
    for cls, idx in raw:
        counters[cls] = counters.get(cls, 0) + 1
        info = CLASS_INFO[cls]
        P = points[idx]
        lo = np.percentile(P, 1.0, axis=0)
        hi = np.percentile(P, 99.0, axis=0)
        k = len(instances)
        inst_of_point[idx] = k
        instances.append({
            "id": f"{info['prefix']}-{counters[cls]:02d}",
            "class": cls,
            "role": info["role"],
            "n_points": int(len(idx)),
            "confidence": round(float(conf[idx].mean()), 3),
            "centroid": np.round(P.mean(0), 3).tolist(),
            "lo": np.round(lo, 3).tolist(),
            "hi": np.round(hi, 3).tolist(),
        })
    # points that voted for a class but ended in a tiny cluster -> background
    point_class = np.where(inst_of_point >= 0, point_class, -1).astype(np.int8)
    return instances, inst_of_point, point_class


# --------------------------------------------------------------------------- 4. structure planes

def _peak(values, lo, hi, bin_size=0.02):
    v = values[(values >= lo) & (values <= hi)]
    if len(v) == 0:
        return (lo + hi) / 2
    bins = np.arange(lo, hi + bin_size, bin_size)
    h, e = np.histogram(v, bins)
    h = ndimage.uniform_filter1d(h.astype(float), 3)
    i = int(h.argmax())
    return float((e[i] + e[i + 1]) / 2)


def detect_structure(points, background_mask, tol=0.07):
    P = points[background_mask]
    mn, mx = np.percentile(P, 0.2, axis=0), np.percentile(P, 99.8, axis=0)
    z0 = _peak(P[:, 2], mn[2] - 0.2, mn[2] + 1.0)
    z1 = _peak(P[:, 2], mx[2] - 1.5, mx[2] + 0.2)
    mid = P[(P[:, 2] > z0 + 0.5) & (P[:, 2] < z1 - 1.0)]
    x0 = _peak(mid[:, 0], mn[0] - 0.2, mn[0] + 1.0)
    x1 = _peak(mid[:, 0], mx[0] - 1.0, mx[0] + 0.2)
    y0 = _peak(mid[:, 1], mn[1] - 0.2, mn[1] + 1.0)
    y1 = _peak(mid[:, 1], mx[1] - 1.0, mx[1] + 0.2)
    room = {"x0": x0, "x1": x1, "y0": y0, "y1": y1, "z0": z0, "z1": z1}

    code = np.full(len(points), -1, np.int8)
    bg = np.flatnonzero(background_mask)
    Q = points[bg]
    c = np.full(len(bg), STRUCT_CODES.index("clutter"), np.int8)
    tests = [
        ("wall_south", np.abs(Q[:, 1] - y0) < tol),
        ("wall_north", np.abs(Q[:, 1] - y1) < tol),
        ("wall_west", np.abs(Q[:, 0] - x0) < tol),
        ("wall_east", np.abs(Q[:, 0] - x1) < tol),
        ("floor", np.abs(Q[:, 2] - z0) < tol),
        ("ceiling", np.abs(Q[:, 2] - z1) < tol),
    ]
    for name, m in tests:
        c[m] = STRUCT_CODES.index(name)
    code[bg] = c
    return {k: round(v, 3) for k, v in room.items()}, code


# --------------------------------------------------------------------------- evaluation vs ground truth

def evaluate(instances, inst_of_point, point_gt, gt_instances):
    for k, inst in enumerate(instances):
        pred = inst_of_point == k
        gts, cnt = np.unique(point_gt[pred], return_counts=True)
        ok = gts >= 0
        if not ok.any():
            inst["gt_name"], inst["iou"] = None, 0.0
            continue
        g = int(gts[ok][cnt[ok].argmax()])
        gtm = point_gt == g
        inter = np.count_nonzero(pred & gtm)
        union = np.count_nonzero(pred | gtm)
        inst["gt_name"] = gt_instances[g]["name"]
        inst["iou"] = round(inter / max(union, 1), 3)
    found = {i["gt_name"] for i in instances}
    missed = [g["name"] for g in gt_instances if g["name"] not in found]
    miou = float(np.mean([i["iou"] for i in instances])) if instances else 0.0
    return {"mean_iou": round(miou, 3), "missed": missed}


def run_segmentation(scan: ScanData, gt_instances: list, cfg: dict, seed: int = 0, progress=None) -> SegmentationResult:
    if progress:
        progress(0.0, "SAM3 마스크 생성 (텍스트 프롬프트: " + ", ".join(CLASS_INFO[c]["prompt"] for c in CLASS_NAMES) + ")")
    det_map, dets = simulate_sam3(scan, gt_instances, cfg, seed)
    if progress:
        progress(0.3, "다시점 투표로 포인트에 라벨 재매핑")
    pclass, conf, seen = vote_point_classes(scan, det_map, dets, cfg["vote_ratio"])
    if progress:
        progress(0.55, "이상점(flying pixel) 제거")
    inlier = remove_outliers(scan.points)
    pclass[~inlier] = -1
    if progress:
        progress(0.7, "3D 클러스터링으로 객체 인스턴스 분리 및 ID 부여")
    instances, inst_of_point, pclass = cluster_instances(
        scan.points, pclass, conf, cfg["cluster_voxel"], cfg["min_instance_points"], cfg.get("min_confidence", 0.6))
    if progress:
        progress(0.85, "미분류 포인트에서 바닥/벽/천장 평면 검출")
    room, struct = detect_structure(scan.points, (inst_of_point < 0) & inlier)
    struct[~inlier] = -1
    stats = evaluate(instances, inst_of_point, scan.point_gt, gt_instances)
    stats.update({
        "n_detections_2d": len(dets),
        "n_points": int(len(scan.points)),
        "n_outliers": int((~inlier).sum()),
        "labelled_ratio": round(float((inst_of_point >= 0).mean()), 3),
        "unseen_points": int((seen == 0).sum()),
    })
    return SegmentationResult(det_map, dets, pclass, inst_of_point, struct, instances, room, stats, inlier)
