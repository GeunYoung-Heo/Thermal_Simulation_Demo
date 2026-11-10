"""Reconstruction -> editable thermal model description ("twin").

Turns detected instances + structure planes into plain dictionaries the user can edit in the app:
    domain        room box from the detected floor / ceiling / wall planes
    devices       controllable thermal devices (with IDs)          -> design variables
    heat_sources  detected equipment without ID                     -> fixed-temperature BC
    doors         detected doors                                    -> BC patch on a wall
    racks         storage racks                                     -> managed zone (+ goods material)
"""
from __future__ import annotations

import copy

import numpy as np

from .common import CLASS_INFO, WALL_FACES

WALL_NAMES = ["wall_south", "wall_north", "wall_west", "wall_east"]


def nearest_wall(xy, room):
    d = {
        "wall_west": abs(xy[0] - room["x0"]),
        "wall_east": abs(room["x1"] - xy[0]),
        "wall_south": abs(xy[1] - room["y0"]),
        "wall_north": abs(room["y1"] - xy[1]),
    }
    return min(d, key=d.get)


def device_box(dev):
    """World AABB of a device dict {x,y,z,yaw,L,T,H}."""
    L, T, H = dev["L"], dev["T"], dev["H"]
    sx, sy = (L, T) if int(dev.get("yaw", 0)) % 180 == 0 else (T, L)
    c = np.array([dev["x"], dev["y"], dev["z"]], float)
    half = np.array([sx, sy, H]) / 2
    return c - half, c + half


def derive_twin(instances: list, room: dict, cfg: dict) -> dict:
    th = cfg["thermal"]
    dev_types = th["device_types"]
    devices, heat_sources, racks, doors = [], [], [], []
    templates = {}
    for inst in instances:
        lo, hi = np.array(inst["lo"]), np.array(inst["hi"])
        c = (lo + hi) / 2
        size = hi - lo
        role = inst["role"]
        if role == "device":
            wall = nearest_wall(c, room)
            yaw = 0 if wall in ("wall_south", "wall_north") else 90
            L, T = (size[0], size[1]) if yaw == 0 else (size[1], size[0])
            spec = dev_types.get(inst["class"], {})
            dev = {
                "id": inst["id"], "class": inst["class"], "enabled": True,
                "x": round(float(c[0]), 3), "y": round(float(c[1]), 3), "z": round(float(c[2]), 3),
                "yaw": yaw, "L": round(float(max(L, 0.2)), 3), "T": round(float(max(T, 0.1)), 3), "H": round(float(max(size[2], 0.2)), 3),
                "T_set": spec.get("T_default", 50.0), "mesh_ref": inst["id"], "wall": wall,
            }
            devices.append(dev)
            if inst["class"] not in templates:
                templates[inst["class"]] = {k: dev[k] for k in ("mesh_ref", "L", "T", "H", "x", "y", "z", "yaw")}
        elif role == "heat_source":
            heat_sources.append({"name": inst["id"], "class": inst["class"],
                                 "lo": np.round(lo, 3).tolist(), "hi": np.round(hi, 3).tolist()})
        elif role == "zone":
            racks.append({"id": inst["id"], "lo": np.round(lo, 3).tolist(), "hi": np.round(hi, 3).tolist(), "in_zone": True})
        elif role == "opening":
            wall = nearest_wall(c, room)
            ax_u = 0 if wall in ("wall_south", "wall_north") else 1
            width = size[ax_u]
            doors.append({
                "name": inst["id"], "wall": wall, "kind": "dock" if width > 2.0 else "personnel",
                "u0": round(float(lo[ax_u]), 3), "u1": round(float(hi[ax_u]), 3),
                "z0": round(float(max(lo[2], room["z0"])), 3), "z1": round(float(hi[2]), 3),
            })
    # device types that were never detected still need a template (size from config)
    for cls, spec in dev_types.items():
        if cls not in templates:
            L, T, H = spec["size"]
            templates[cls] = {"mesh_ref": None, "L": L, "T": T, "H": H}
    return {
        "domain": {"lo": [room["x0"], room["y0"], room["z0"]], "hi": [room["x1"], room["y1"], room["z1"]]},
        "devices": devices,
        "heat_sources": heat_sources,
        "racks": racks,
        "doors": doors,
        "templates": templates,
    }


def default_boundaries(twin: dict, cfg: dict, preset: str = "winter") -> list:
    """BC table rows for a preset: walls/floor/ceiling, each door, each heat source."""
    th = cfg["thermal"]
    p = th["presets"][preset]["boundaries"]
    rows = []
    labels = {"wall_south": "남측 외벽", "wall_north": "북측 외벽", "wall_west": "서측 외벽",
              "wall_east": "동측 외벽", "ceiling": "지붕", "floor": "바닥(지중)"}
    for name in ["wall_south", "wall_north", "wall_west", "wall_east", "ceiling", "floor"]:
        b = p[name]
        rows.append({"name": name, "label": labels[name], "kind": "외피", "type": b["type"], "T": b["T"], "U": b["U"]})
    for d in twin["doors"]:
        b = p["door_dock"] if d["kind"] == "dock" else p["door_personnel"]
        lab = ("도크 도어" if d["kind"] == "dock" else "출입문") + f" ({d['name']})"
        rows.append({"name": d["name"], "label": lab, "kind": "출입문", "type": b["type"], "T": b["T"], "U": b["U"]})
    for h in twin["heat_sources"]:
        b = th["heat_source_defaults"].get(h["class"], {"type": "fixed", "T": 30.0})
        rows.append({"name": h["name"], "label": CLASS_INFO[h["class"]]["label"] + f" ({h['name']})",
                     "kind": "발열설비", "type": b["type"], "T": b["T"], "U": None})
    return rows


def preset_devices(twin: dict, cfg: dict, preset: str = "winter") -> list:
    active = set(cfg["thermal"]["presets"][preset]["active_device_classes"])
    types = cfg["thermal"]["device_types"]
    devs = copy.deepcopy(twin["devices"])
    for d in devs:
        d["enabled"] = d["class"] in active
        d["T_set"] = types.get(d["class"], {}).get("T_default", d["T_set"])
    return devs


def new_device(twin: dict, cfg: dict, cls: str, existing_ids, wall="wall_north", u=None) -> dict:
    """A new device of class `cls` mounted on `wall` at along-wall coordinate u (default: wall middle)."""
    spec = cfg["thermal"]["device_types"][cls]
    tpl = twin["templates"][cls]
    lo, hi = np.array(twin["domain"]["lo"]), np.array(twin["domain"]["hi"])
    prefix = CLASS_INFO[cls]["prefix"]
    n = 1
    while f"{prefix}-{n:02d}" in existing_ids:
        n += 1
    L, T, H = tpl["L"], tpl["T"], tpl["H"]
    ax, side = WALL_FACES[wall]
    yaw = 0 if ax == 1 else 90
    along = 0 if ax == 1 else 1
    if u is None:
        u = (lo[along] + hi[along]) / 2
    off = spec.get("wall_offset", 0.05) + T / 2
    pos = np.zeros(3)
    pos[along] = u
    pos[ax] = lo[ax] + off if side == 0 else hi[ax] - off
    pos[2] = lo[2] + spec.get("mount_z", 0.15) + H / 2
    return {"id": f"{prefix}-{n:02d}", "class": cls, "enabled": True,
            "x": round(float(pos[0]), 3), "y": round(float(pos[1]), 3), "z": round(float(pos[2]), 3),
            "yaw": yaw, "L": L, "T": T, "H": H, "T_set": spec.get("T_default", 50.0),
            "mesh_ref": tpl.get("mesh_ref"), "wall": wall}
