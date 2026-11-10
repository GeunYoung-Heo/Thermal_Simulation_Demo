"""Saving / loading of pipeline outputs and simulation results.

data/                         (pipeline outputs, produced by scripts/build_twin.py)
    scan.npz                  fused point cloud, station poses, panorama RGB images
    scan_depth.npz            panorama depth maps (+ ground-truth labels for evaluation)
    segmentation.npz/.json    SAM3 masks, point labels, detected instances, room planes
    mesh_display.npz          decimated mesh parts for the viewer
    mesh_parts.json           part names / classes / colours
    mesh_full.ply             full-resolution labelled mesh (open in MeshLab / CloudCompare)
    twin.json                 thermal model description derived from the reconstruction
results/
    <timestamp>_<label>.npz   simulation result: 3D temperature field + grid + metadata (JSON)
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np

from .scanner import PanoGeometry, ScanData
from .segmentation import SegmentationResult
from .thermal import Grid, SimResult


def _dump(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=_np_default)


def _np_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------------ scan

def save_scan(scan: ScanData, d: Path):
    g = scan.geom
    np.savez_compressed(d / "scan.npz", points=scan.points, colors=scan.colors, point_station=scan.point_station,
                        point_gt=scan.point_gt, positions=scan.positions, rotations=scan.rotations, rgb=scan.rgb,
                        geom=np.array([g.width, g.height, g.el_min, g.el_max]))
    np.savez_compressed(d / "scan_depth.npz", depth=scan.depth.astype(np.float16), gt_label=scan.gt_label)


def load_scan(d: Path, with_depth=False) -> ScanData:
    z = np.load(d / "scan.npz")
    w, h, e0, e1 = z["geom"]
    g = PanoGeometry(int(w), int(h), float(e0), float(e1))
    depth = gt = None
    if with_depth:
        zd = np.load(d / "scan_depth.npz")
        depth = zd["depth"].astype(np.float32)
        gt = zd["gt_label"]
    return ScanData(geom=g, positions=z["positions"], rotations=z["rotations"], rgb=z["rgb"], depth=depth,
                    gt_label=gt, points=z["points"], colors=z["colors"], point_station=z["point_station"],
                    point_gt=z["point_gt"])


# ------------------------------------------------------------------ segmentation

def save_segmentation(seg: SegmentationResult, d: Path):
    np.savez_compressed(d / "segmentation.npz", det_map=seg.det_map, point_class=seg.point_class,
                        point_instance=seg.point_instance, point_struct=seg.point_struct, inlier=seg.inlier)
    _dump({"detections": seg.detections, "instances": seg.instances, "room": seg.room, "stats": seg.stats},
          d / "segmentation.json")


def load_segmentation(d: Path) -> SegmentationResult:
    z = np.load(d / "segmentation.npz")
    j = load_json(d / "segmentation.json")
    return SegmentationResult(det_map=z["det_map"], detections=j["detections"], point_class=z["point_class"],
                              point_instance=z["point_instance"], point_struct=z["point_struct"],
                              instances=j["instances"], room=j["room"], stats=j["stats"], inlier=z["inlier"])


# ------------------------------------------------------------------ mesh

def save_mesh(parts: list, full: dict, stats: dict, d: Path):
    arrs = {}
    meta = []
    for i, p in enumerate(parts):
        arrs[f"v{i}"] = p["verts"].astype(np.float32)
        arrs[f"f{i}"] = p["faces"].astype(np.int32)
        arrs[f"c{i}"] = p["vcolor"].astype(np.uint8)
        meta.append({k: v for k, v in p.items() if k not in ("verts", "faces", "vcolor")})
    np.savez_compressed(d / "mesh_display.npz", **arrs)
    _dump({"parts": meta, "stats": stats}, d / "mesh_parts.json")
    write_ply(d / "mesh_full.ply", full["verts"], full["faces"], full["face_part"], full["names"])


def load_mesh(d: Path):
    z = np.load(d / "mesh_display.npz")
    j = load_json(d / "mesh_parts.json")
    parts = []
    for i, m in enumerate(j["parts"]):
        parts.append({**m, "verts": z[f"v{i}"], "faces": z[f"f{i}"], "vcolor": z[f"c{i}"]})
    return parts, j["stats"]


def write_ply(path, verts, faces, face_part, names):
    """Binary PLY; vertex colour = part colour, face property 'part' = part index (see header comments)."""
    def hex2rgb(h):
        h = h.lstrip("#")
        return [int(h[i:i + 2], 16) for i in (0, 2, 4)]
    pal = np.array([hex2rgb(n["color"]) for n in names], np.uint8)
    vpart = np.zeros(len(verts), np.int64)
    vpart[faces[:, 0]] = face_part
    vcol = pal[np.clip(vpart, 0, len(pal) - 1)]
    header = ["ply", "format binary_little_endian 1.0"]
    header += [f"comment part {i} {n['key']}" for i, n in enumerate(names)]
    header += [f"element vertex {len(verts)}", "property float x", "property float y", "property float z",
               "property uchar red", "property uchar green", "property uchar blue",
               f"element face {len(faces)}", "property list uchar int vertex_indices", "property short part",
               "end_header"]
    vdt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    va = np.empty(len(verts), vdt)
    va["x"], va["y"], va["z"] = verts[:, 0], verts[:, 1], verts[:, 2]
    va["r"], va["g"], va["b"] = vcol[:, 0], vcol[:, 1], vcol[:, 2]
    fdt = np.dtype([("n", "u1"), ("i", "<i4", 3), ("p", "<i2")])
    fa = np.empty(len(faces), fdt)
    fa["n"] = 3
    fa["i"] = faces
    fa["p"] = face_part
    with open(path, "wb") as f:
        f.write(("\n".join(header) + "\n").encode("ascii"))
        f.write(va.tobytes())
        f.write(fa.tobytes())


# ------------------------------------------------------------------ twin

def save_twin(twin: dict, d: Path):
    _dump(twin, d / "twin.json")


def load_twin(d: Path) -> dict:
    return load_json(d / "twin.json")


# ------------------------------------------------------------------ simulation results

def save_result(res: SimResult, results_dir: Path, label: str = "result") -> Path:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^\w\-가-힣]+", "_", label).strip("_") or "result"
    path = results_dir / f"{time.strftime('%Y%m%d_%H%M%S')}_{safe}.npz"
    g = res.grid
    meta = dict(res.meta)
    meta.update({"saved_label": label, "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                 "devices": res.devices, "bcs": res.bcs, "groups": g.groups,
                 "body_names": g.body_names, "body_kind": g.body_kind})
    np.savez_compressed(path, T=res.T.astype(np.float32), cell_type=g.cell_type, body=g.body, zone=g.zone,
                        origin=g.origin, h=np.array(g.h), meta=np.array(json.dumps(meta, ensure_ascii=False, default=_np_default)))
    return path


def load_result(path) -> SimResult:
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    shape = z["T"].shape
    g = Grid(origin=z["origin"], h=float(z["h"]), shape=shape, cell_type=z["cell_type"], body=z["body"],
             body_names=meta.get("body_names", []), body_kind=meta.get("body_kind", []), zone=z["zone"],
             k=np.zeros(0, np.float32), groups=meta.get("groups", []), face_group={})
    return SimResult(T=z["T"], grid=g, devices=meta.get("devices", []), bcs=meta.get("bcs", []), meta=meta)


def list_results(results_dir: Path):
    results_dir = Path(results_dir)
    if not results_dir.exists():
        return []
    return sorted(results_dir.glob("*.npz"), reverse=True)
