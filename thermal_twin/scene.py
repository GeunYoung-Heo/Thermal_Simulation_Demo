"""Virtual 'real world' warehouse built from axis-aligned boxes.

Only the sensor simulators (scanner / SAM3) look at this ground truth.  Everything downstream
works on the reconstructed data, exactly like it would with a real scan.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from .common import CLASS_NAMES

# material / texture codes used by the scanner's colour function
MAT_PLAIN, MAT_FLOOR, MAT_WALL, MAT_CEILING, MAT_ROLLUP, MAT_CARDBOARD, MAT_WOOD = range(7)


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class Scene:
    name: str
    room: np.ndarray                 # (3,) interior size
    lo: np.ndarray                   # (N,3) box min corners
    hi: np.ndarray                   # (N,3) box max corners
    color: np.ndarray                # (N,3) base colour 0..1
    material: np.ndarray             # (N,)  texture code
    inst: np.ndarray                 # (N,)  instance index, -1 = background
    instances: list                  # [{name, class, class_id}]
    stations: np.ndarray             # (M,3) scanner positions
    floor_lines: list = field(default_factory=list)
    cfg: dict = field(default_factory=dict)

    @property
    def n_boxes(self) -> int:
        return len(self.lo)


class _Builder:
    def __init__(self, room):
        self.room = np.asarray(room, float)
        self.lo, self.hi, self.color, self.mat, self.inst = [], [], [], [], []
        self.instances = []

    def new_instance(self, name, cls):
        self.instances.append({"name": name, "class": cls, "class_id": CLASS_NAMES.index(cls)})
        return len(self.instances) - 1

    def box(self, lo, hi, color, mat=MAT_PLAIN, inst=-1):
        lo = np.asarray(lo, float)
        hi = np.asarray(hi, float)
        a, b = np.minimum(lo, hi), np.maximum(lo, hi)
        self.lo.append(a)
        self.hi.append(b)
        self.color.append(np.asarray(color, float))
        self.mat.append(mat)
        self.inst.append(inst)

    # local wall frame:  u = along wall, v = distance from wall into the room, z = height
    def wall_box(self, wall, u0, u1, v0, v1, z0, z1, color, mat=MAT_PLAIN, inst=-1):
        Lx, Ly, _ = self.room
        if wall == "wall_south":
            lo, hi = (u0, v0, z0), (u1, v1, z1)
        elif wall == "wall_north":
            lo, hi = (u0, Ly - v1, z0), (u1, Ly - v0, z1)
        elif wall == "wall_west":
            lo, hi = (v0, u0, z0), (v1, u1, z1)
        elif wall == "wall_east":
            lo, hi = (Lx - v1, u0, z0), (Lx - v0, u1, z1)
        else:
            raise ValueError(wall)
        self.box(lo, hi, color, mat, inst)


WHITE = (0.93, 0.93, 0.92)
RACK_BLUE = (0.12, 0.30, 0.62)
RACK_ORANGE = (0.93, 0.45, 0.10)
WOOD = (0.62, 0.47, 0.30)
CARDBOARD = [(0.72, 0.56, 0.38), (0.66, 0.50, 0.33), (0.80, 0.66, 0.47), (0.88, 0.88, 0.86), (0.55, 0.63, 0.75)]


def _add_rack(b: _Builder, spec, levels, upright_h, empty_ratio, rng):
    inst = b.new_instance(spec["name"], "pallet_rack")
    x0, bays, bw, y0, sides = spec["x0"], spec["bays"], spec["bay_width"], spec["y0"], spec["sides"]
    side_d, gap = 1.1, 0.2
    for s in range(sides):
        ys0 = y0 + s * (side_d + gap)
        ys1 = ys0 + side_d
        # uprights
        for k in range(bays + 1):
            xk = x0 + k * bw
            for yy in (ys0, ys1 - 0.08):
                b.box((xk - 0.04, yy, 0.0), (xk + 0.04, yy + 0.08, upright_h), RACK_BLUE, inst=inst)
            # bracing strip between front / back upright
            for zb in np.arange(0.8, upright_h, 1.6):
                b.box((xk - 0.02, ys0 + 0.08, zb), (xk + 0.02, ys1 - 0.08, zb + 0.05), RACK_BLUE, inst=inst)
        # beams
        for lz in levels[1:]:
            for k in range(bays):
                xa, xb = x0 + k * bw + 0.04, x0 + (k + 1) * bw - 0.04
                for yy in (ys0, ys1 - 0.05):
                    b.box((xa, yy, lz - 0.12), (xb, yy + 0.05, lz), RACK_ORANGE, inst=inst)
        # pallets + goods
        for li, lz in enumerate(levels):
            base = lz
            top = levels[li + 1] - 0.12 if li + 1 < len(levels) else upright_h + 0.4
            for k in range(bays):
                for p in range(2):
                    if rng.random() < empty_ratio:
                        continue
                    px0 = x0 + k * bw + 0.10 + p * 1.27
                    b.box((px0, ys0 + 0.05, base), (px0 + 1.2, ys0 + 1.05, base + 0.144), WOOD, MAT_WOOD, inst)
                    gmax = min(1.35, top - base - 0.144 - 0.08)
                    gh = rng.uniform(0.6, gmax)
                    sh = rng.uniform(0.02, 0.08, 4)
                    c = CARDBOARD[rng.integers(len(CARDBOARD))]
                    g0 = base + 0.144
                    if rng.random() < 0.3:      # two stacked cartons
                        h1 = gh * rng.uniform(0.4, 0.6)
                        b.box((px0 + sh[0], ys0 + 0.05 + sh[1], g0), (px0 + 1.2 - sh[2], ys0 + 1.05 - sh[3], g0 + h1), c, MAT_CARDBOARD, inst)
                        c2 = CARDBOARD[rng.integers(len(CARDBOARD))]
                        s2 = rng.uniform(0.05, 0.15, 4)
                        b.box((px0 + s2[0], ys0 + 0.05 + s2[1], g0 + h1), (px0 + 1.2 - s2[2], ys0 + 1.05 - s2[3], g0 + gh), c2, MAT_CARDBOARD, inst)
                    else:
                        b.box((px0 + sh[0], ys0 + 0.05 + sh[1], g0), (px0 + 1.2 - sh[2], ys0 + 1.05 - sh[3], g0 + gh), c, MAT_CARDBOARD, inst)


def _add_radiator(b: _Builder, wall, center, z0, inst, size=(1.6, 0.12, 0.6), offset=0.06):
    L, T, H = size
    u0, u1 = center - L / 2, center + L / 2
    b.wall_box(wall, u0, u1, offset, offset + 0.025, z0, z0 + H, WHITE, inst=inst)          # back plate
    pitch = 0.08
    for k in range(int(L / pitch)):
        ua = u0 + 0.015 + k * pitch
        b.wall_box(wall, ua, ua + 0.05, offset + 0.025, offset + T, z0 + 0.03, z0 + H - 0.03, WHITE, inst=inst)
    b.wall_box(wall, u0, u1, offset + 0.025, offset + T, z0 + H - 0.03, z0 + H, (0.88, 0.88, 0.87), inst=inst)  # top grille
    for ue in (u0 + 0.06, u1 - 0.09):     # supply / return pipes
        b.wall_box(wall, ue, ue + 0.03, offset + 0.03, offset + 0.06, 0.0, z0 + 0.05, (0.75, 0.75, 0.74), inst=inst)


def _add_air_conditioner(b: _Builder, wall, center, z0, inst):
    b.wall_box(wall, center - 0.65, center + 0.65, 0.0, 0.45, z0, z0 + 0.55, (0.92, 0.92, 0.93), inst=inst)
    b.wall_box(wall, center - 0.55, center + 0.55, 0.45, 0.47, z0 + 0.08, z0 + 0.47, (0.45, 0.47, 0.50), inst=inst)
    for ue in (center - 0.6, center + 0.5):
        b.wall_box(wall, ue, ue + 0.1, 0.0, 0.3, z0 - 0.12, z0, (0.55, 0.55, 0.55), inst=inst)


def _add_door(b: _Builder, spec, inst):
    w, h, a, wall = spec["width"], spec["height"], spec["center"], spec["wall"]
    if spec.get("kind") == "dock":
        b.wall_box(wall, a - w / 2, a + w / 2, 0.0, 0.05, 0.0, h, (0.56, 0.61, 0.68), MAT_ROLLUP, inst)
        for ue in (a - w / 2 - 0.15, a + w / 2):
            b.wall_box(wall, ue, ue + 0.15, 0.0, 0.10, 0.0, h + 0.15, (0.30, 0.30, 0.32), inst=inst)
            b.wall_box(wall, ue, ue + 0.15, 0.0, 0.18, 0.15, 0.6, (0.08, 0.08, 0.08), inst=inst)   # dock bumper
        b.wall_box(wall, a - w / 2 - 0.15, a + w / 2 + 0.15, 0.0, 0.25, h + 0.15, h + 0.55, (0.40, 0.42, 0.45), inst=inst)  # roll housing
    else:
        b.wall_box(wall, a - w / 2, a + w / 2, 0.0, 0.04, 0.0, h, (0.70, 0.28, 0.22), inst=inst)
        b.wall_box(wall, a - w / 2 - 0.08, a + w / 2 + 0.08, 0.0, 0.06, h, h + 0.08, (0.35, 0.35, 0.36), inst=inst)
        b.wall_box(wall, a + w / 2 - 0.15, a + w / 2 - 0.05, 0.04, 0.09, 1.0, 1.05, (0.75, 0.75, 0.75), inst=inst)  # handle


def _add_pallet_stack(b: _Builder, x, y, rng):
    b.box((x - 0.6, y - 0.5, 0.0), (x + 0.6, y + 0.5, 0.144), WOOD, MAT_WOOD)
    gh = rng.uniform(0.6, 1.5)
    c = CARDBOARD[rng.integers(len(CARDBOARD))]
    b.box((x - 0.55, y - 0.45, 0.144), (x + 0.55, y + 0.45, 0.144 + gh), c, MAT_CARDBOARD)


def _add_forklift(b: _Builder, x, y):
    body = (0.95, 0.62, 0.10)
    dark = (0.12, 0.12, 0.13)
    b.box((x - 1.0, y - 0.55, 0.25), (x + 0.6, y + 0.55, 1.15), body)
    b.box((x - 1.25, y - 0.5, 0.3), (x - 1.0, y + 0.5, 1.05), (0.25, 0.25, 0.27))     # counterweight
    for xx in (x - 0.7, x + 0.35):
        for yy in (y - 0.55, y + 0.4):
            b.box((xx - 0.22, yy, 0.0), (xx + 0.22, yy + 0.15, 0.45), dark)
    for yy in (y - 0.5, y + 0.45):
        b.box((x - 0.8, yy, 1.15), (x - 0.75, yy + 0.05, 2.15), dark)
        b.box((x + 0.35, yy, 1.15), (x + 0.4, yy + 0.05, 2.15), dark)
    b.box((x - 0.82, y - 0.52, 2.15), (x + 0.42, y + 0.52, 2.2), dark)                # overhead guard
    b.box((x + 0.6, y - 0.4, 0.05), (x + 0.7, y + 0.4, 2.5), (0.3, 0.3, 0.32))         # mast
    for yy in (y - 0.3, y + 0.2):
        b.box((x + 0.7, yy, 0.05), (x + 1.9, yy + 0.1, 0.1), (0.3, 0.3, 0.32))         # forks


def build_scene(cfg: dict) -> Scene:
    rng = np.random.default_rng(cfg.get("seed", 0))
    room = np.asarray(cfg["room"]["size"], float)
    Lx, Ly, Lz = room
    t = cfg["room"].get("wall_thickness", 0.3)
    b = _Builder(room)

    # structure: boxes outside the interior volume, interior faces are what the scanner sees
    b.box((-t, -t, -t), (Lx + t, Ly + t, 0.0), (0.66, 0.68, 0.66), MAT_FLOOR)
    b.box((-t, -t, Lz), (Lx + t, Ly + t, Lz + t), (0.82, 0.82, 0.83), MAT_CEILING)
    b.box((-t, -t, 0.0), (0.0, Ly + t, Lz), (0.88, 0.89, 0.90), MAT_WALL)
    b.box((Lx, -t, 0.0), (Lx + t, Ly + t, Lz), (0.88, 0.89, 0.90), MAT_WALL)
    b.box((0.0, -t, 0.0), (Lx, 0.0, Lz), (0.88, 0.89, 0.90), MAT_WALL)
    b.box((0.0, Ly, 0.0), (Lx, Ly + t, Lz), (0.88, 0.89, 0.90), MAT_WALL)
    # roof trusses (unlabelled clutter near the ceiling)
    for xt in np.arange(3.0, Lx, 6.0):
        b.box((xt - 0.1, 0.0, Lz - 0.45), (xt + 0.1, Ly, Lz), (0.62, 0.63, 0.66))

    for d in cfg.get("doors", []):
        _add_door(b, d, b.new_instance(d["name"], "door"))

    for r in cfg.get("racks", []):
        _add_rack(b, r, cfg["rack_levels"], cfg["rack_upright_height"], cfg.get("rack_empty_ratio", 0.1), rng)

    dev_types = cfg["thermal"]["device_types"]
    counters = {}
    for d in cfg.get("devices", []):
        cls = d["class"]
        counters[cls] = counters.get(cls, 0) + 1
        inst = b.new_instance(f"{cls}_{counters[cls]}", cls)
        if cls == "radiator":
            spec = dev_types["radiator"]
            _add_radiator(b, d["wall"], d["center"], d.get("z0", spec["mount_z"]), inst, tuple(spec["size"]), spec["wall_offset"])
        elif cls == "air_conditioner":
            _add_air_conditioner(b, d["wall"], d["center"], d.get("z0", 4.4), inst)

    for e in cfg.get("equipment", []):
        inst = b.new_instance(e["name"], e["class"])
        c, s = np.asarray(e["center"]), np.asarray(e["size"])
        if e["class"] == "battery_charger":
            b.box(c - s / 2, c + s / 2, (0.86, 0.72, 0.16), inst=inst)
            b.box(c + np.array([-s[0] / 2 - 0.02, -0.2, 0.2]), c + np.array([-s[0] / 2, 0.2, 0.5]), (0.1, 0.12, 0.15), inst=inst)
        else:
            b.box(c - s / 2, c + s / 2, (0.68, 0.70, 0.72), inst=inst)
            b.box(c + np.array([s[0] / 2, -0.02, -s[2] / 2 + 0.1]), c + np.array([s[0] / 2 + 0.02, 0.02, s[2] / 2 - 0.1]), (0.35, 0.35, 0.36), inst=inst)

    cl = cfg.get("clutter", {})
    for x, y in cl.get("forklifts", []):
        _add_forklift(b, x, y)
    for x, y in cl.get("loose_pallets", []):
        _add_pallet_stack(b, x, y, rng)
    lights = cl.get("ceiling_lights", {})
    for yl in lights.get("rows", []):
        for xl in lights.get("xs", []):
            b.box((xl - 0.6, yl - 0.2, Lz - 0.62), (xl + 0.6, yl + 0.2, Lz - 0.5), (0.97, 0.97, 0.95))
            b.box((xl - 0.02, yl - 0.02, Lz - 0.5), (xl + 0.02, yl + 0.02, Lz), (0.4, 0.4, 0.4))
    for wall, u in cl.get("extinguishers", []):
        b.wall_box(wall, u - 0.09, u + 0.09, 0.02, 0.2, 0.9, 1.45, (0.80, 0.10, 0.10))

    sc = cfg["scanner"]
    zs = sc.get("station_height", 1.6)
    stations = np.array([[x, y, zs] for x, y in sc["stations"]], float)

    return Scene(
        name=cfg.get("name", "scene"),
        room=room,
        lo=np.array(b.lo),
        hi=np.array(b.hi),
        color=np.array(b.color),
        material=np.array(b.mat, np.int8),
        inst=np.array(b.inst, np.int32),
        instances=b.instances,
        stations=stations,
        floor_lines=cfg.get("floor_lines", []),
        cfg=cfg,
    )
