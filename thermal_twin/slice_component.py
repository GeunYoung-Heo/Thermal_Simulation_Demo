"""Streamlit custom component: browser-side section viewer.

Streamlit sliders only report their value on release and every change reruns the script (the plot is
re-sent and flickers).  This component keeps the axis buttons, position slider and colour range inside
the browser: the 2D heatmap is updated on every slider 'input' event and the 3D slice plane is moved in
place.  The temperature field is shipped once as base64 float32.
"""
from __future__ import annotations

import base64
import hashlib
import json
import shutil
from collections import OrderedDict
from pathlib import Path

import numpy as np
import streamlit.components.v1 as components

from . import viz
from .common import CLASS_INFO, TEMP_COLORSCALE
from .twin import device_box

_DIR = Path(__file__).resolve().parent / "slice_viewer"


def _ensure_plotly_js():
    """Serve plotly.js from the installed plotly package (works offline, matches plotly.py's JSON)."""
    import plotly
    src = Path(plotly.__file__).parent / "package_data" / "plotly.min.js"
    dst = _DIR / "plotly.min.js"
    if src.exists() and (not dst.exists() or dst.stat().st_size != src.stat().st_size):
        shutil.copyfile(src, dst)


_ensure_plotly_js()
_component = components.declare_component("thermal_slice_viewer", path=str(_DIR))
_CACHE: "OrderedDict[str, dict]" = OrderedDict()


def _spec(T, origin, h, low, high, twin, devices, parts, original_pose, height):
    on = [d for d in devices if d.get("enabled", True)]
    T32 = np.ascontiguousarray(T, dtype=np.float32)
    racks = [{"lo": list(map(float, r["lo"])), "hi": list(map(float, r["hi"]))} for r in twin["racks"] if r.get("in_zone", True)]
    heat = [{"lo": list(map(float, x["lo"])), "hi": list(map(float, x["hi"]))} for x in twin["heat_sources"]]
    devs = []
    for d in on:
        lo, hi = device_box(d)
        devs.append({"id": d["id"], "lo": lo.tolist(), "hi": hi.tolist(), "T": float(d["T_set"]),
                     "top": float(d["z"] + d["H"] / 2), "color": CLASS_INFO[d["class"]]["color"]})
    key_blob = json.dumps([racks, heat, devs, float(low), float(high), float(h), list(map(float, origin)), int(height)])
    data_id = hashlib.md5(T32.tobytes() + key_blob.encode()).hexdigest()
    if data_id in _CACHE:
        _CACHE.move_to_end(data_id)
        return _CACHE[data_id]
    zc = origin[2] + (np.arange(T.shape[2]) + 0.5) * h
    zi = int(np.argmin(np.abs(zc - 1.6)))
    fig3d = viz.fig_slice3d(T, origin, h, 2, zi, low, high, None, twin, on, parts, original_pose, height=height - 56)
    mid = 0.5 * (low + high)
    spec = {
        "data_id": data_id, "height": int(height),
        "T": base64.b64encode(T32.tobytes()).decode("ascii"), "shape": list(T32.shape),
        "origin": [float(v) for v in origin], "h": float(h), "low": float(low), "high": float(high),
        "crange": [mid - 15.0, mid + 15.0], "colorscale": TEMP_COLORSCALE,
        "racks": racks, "heat": heat, "devices": devs,
        "fig3d": json.loads(fig3d.to_json()),
    }
    _CACHE[data_id] = spec
    while len(_CACHE) > 6:
        _CACHE.popitem(last=False)
    return spec


def slice_viewer(T, origin, h, low, high, twin, devices, parts=None, original_pose=None, key=None, height=640):
    spec = _spec(T, np.asarray(origin, float), h, low, high, twin, devices, parts, original_pose, height)
    return _component(spec=spec, key=key, default=None)
