"""Shared constants: object classes, roles, colours."""

# Object classes that the (simulated) SAM3 model is prompted with.
#   role = device       -> controllable thermal device, gets an ID (RAD-01 ...), design variable
#   role = heat_source  -> non-controllable equipment, used as a boundary condition (fixed temperature)
#   role = zone         -> storage racks: define the managed temperature zone
#   role = opening      -> doors: boundary-condition patch on a wall
CLASS_INFO = {
    "radiator": {
        "role": "device", "prefix": "RAD", "prompt": "radiator",
        "label": "라디에이터", "color": "#eb6834",
    },
    "air_conditioner": {
        "role": "device", "prefix": "AC", "prompt": "wall-mounted air conditioner",
        "label": "냉방기(유닛쿨러)", "color": "#2a78d6",
    },
    "battery_charger": {
        "role": "heat_source", "prefix": "CHG", "prompt": "forklift battery charger",
        "label": "지게차 충전기", "color": "#eda100",
    },
    "electrical_cabinet": {
        "role": "heat_source", "prefix": "EP", "prompt": "electrical cabinet",
        "label": "전기 분전반", "color": "#e87ba4",
    },
    "pallet_rack": {
        "role": "zone", "prefix": "RACK", "prompt": "pallet rack with goods",
        "label": "팔레트 랙(보관구역)", "color": "#1baf7a",
    },
    "door": {
        "role": "opening", "prefix": "DOOR", "prompt": "door",
        "label": "출입문/도크도어", "color": "#6250d6",
    },
}

CLASS_NAMES = list(CLASS_INFO.keys())          # index in this list = class id (0..5)
BACKGROUND = -1

# Structure (un-detected surfaces) found by plane detection.  They carry no ID and become BCs.
STRUCTURE_PARTS = {
    "floor": "바닥",
    "ceiling": "지붕/천장",
    "wall_south": "남측 외벽",
    "wall_north": "북측 외벽",
    "wall_west": "서측 외벽",
    "wall_east": "동측 외벽",
    "clutter": "기타(미분류)",
}

STRUCTURE_COLORS = {
    "floor": "#b9bcb6",
    "ceiling": "#d6d6d2",
    "wall_south": "#cfd3d6",
    "wall_north": "#cfd3d6",
    "wall_west": "#c7cbce",
    "wall_east": "#c7cbce",
    "clutter": "#8d8c86",
}

# wall name -> (axis, side)   side 0 = low coordinate face, 1 = high coordinate face
WALL_FACES = {
    "wall_west": (0, 0),
    "wall_east": (0, 1),
    "wall_south": (1, 0),
    "wall_north": (1, 1),
    "floor": (2, 0),
    "ceiling": (2, 1),
}

# Diverging temperature colour scale (blue <-> neutral grey <-> red).
TEMP_COLORSCALE = [
    [0.00, "#0d366b"],
    [0.15, "#1c5cab"],
    [0.30, "#3987e5"],
    [0.42, "#9ec5f4"],
    [0.50, "#f0efec"],
    [0.58, "#f4b4a6"],
    [0.70, "#e8705f"],
    [0.85, "#c73a35"],
    [1.00, "#7d1a1a"],
]
