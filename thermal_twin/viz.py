"""Plotly figures for the Streamlit app (no Streamlit code here, so they can be reused in notebooks)."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from .common import CLASS_INFO, STRUCTURE_COLORS, STRUCTURE_PARTS, TEMP_COLORSCALE
from .segmentation import STRUCT_CODES
from .twin import device_box

INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
DEFAULT_CAMERA = dict(eye=dict(x=-0.95, y=-1.55, z=1.05), center=dict(x=0, y=0, z=-0.15))
AXIS_NAMES = ["x", "y", "z"]
AXIS_LABELS = ["x (서→동) [m]", "y (남→북) [m]", "z (높이) [m]"]


def _rgb(c):
    c = np.asarray(c, np.uint8)
    return [f"rgb({r},{g},{b})" for r, g, b in c]


def _hex2rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], np.uint8)


def _scene_layout(fig, height=640, camera=None, legend=True):
    ax = dict(showbackground=False, gridcolor=GRID, zerolinecolor=GRID, color=INK2)
    fig.update_layout(
        scene=dict(aspectmode="data", camera=camera or DEFAULT_CAMERA,
                   xaxis=dict(title="x [m]", **ax), yaxis=dict(title="y [m]", **ax), zaxis=dict(title="z [m]", **ax)),
        height=height, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)",
        showlegend=legend, legend=dict(itemsizing="constant", bgcolor="rgba(255,255,255,0.7)", font=dict(color=INK)),
        uirevision="keep",
    )
    return fig


# =============================================================================== point clouds

def _subsample(n, max_points, seed=0):
    if n <= max_points:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, max_points, replace=False))


def fig_pointcloud(points, colors, max_points=180_000, z_max=None, stations=None, outlier_mask=None):
    keep = np.ones(len(points), bool) if z_max is None else points[:, 2] < z_max
    idx = np.flatnonzero(keep)
    idx = idx[_subsample(len(idx), max_points)]
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=points[idx, 0], y=points[idx, 1], z=points[idx, 2], mode="markers",
                               marker=dict(size=1.3, color=_rgb(colors[idx])), name="포인트클라우드", hoverinfo="skip"))
    if outlier_mask is not None:
        oi = np.flatnonzero(outlier_mask & keep)
        oi = oi[_subsample(len(oi), 25_000)]
        fig.add_trace(go.Scatter3d(x=points[oi, 0], y=points[oi, 1], z=points[oi, 2], mode="markers",
                                   marker=dict(size=1.6, color="#e34948"), name="이상점(flying pixel)", hoverinfo="skip"))
    if stations is not None:
        fig.add_trace(go.Scatter3d(x=stations[:, 0], y=stations[:, 1], z=stations[:, 2], mode="markers+text",
                                   marker=dict(size=5, color=INK, symbol="diamond"),
                                   text=[f"S{i + 1}" for i in range(len(stations))], textposition="top center",
                                   textfont=dict(color=INK, size=11), name="스캔 위치"))
    return _scene_layout(fig)


def fig_labeled_pointcloud(points, point_instance, point_struct, instances, max_points=180_000, z_max=None):
    keep = np.ones(len(points), bool) if z_max is None else points[:, 2] < z_max
    fig = go.Figure()
    # structure (no ID) in greys
    for si, name in enumerate(STRUCT_CODES):
        m = np.flatnonzero((point_instance < 0) & (point_struct == si) & keep)
        m = m[_subsample(len(m), max_points // 8)]
        if len(m) == 0:
            continue
        fig.add_trace(go.Scatter3d(x=points[m, 0], y=points[m, 1], z=points[m, 2], mode="markers",
                                   marker=dict(size=1.2, color=STRUCTURE_COLORS[name]), name=f"{name} (ID 없음)",
                                   hoverinfo="skip", legendgroup="struct"))
    by_class = {}
    for k, inst in enumerate(instances):
        by_class.setdefault(inst["class"], []).append(k)
    for cls, ks in by_class.items():
        m = np.flatnonzero(np.isin(point_instance, ks) & keep)
        m = m[_subsample(len(m), max_points // 2)]
        fig.add_trace(go.Scatter3d(x=points[m, 0], y=points[m, 1], z=points[m, 2], mode="markers",
                                   marker=dict(size=1.6, color=CLASS_INFO[cls]["color"]),
                                   name=CLASS_INFO[cls]["label"], hoverinfo="skip"))
    tx = [inst["id"] for inst in instances]
    c = np.array([[(i["lo"][0] + i["hi"][0]) / 2, (i["lo"][1] + i["hi"][1]) / 2, min(i["hi"][2] + 0.3, 6.6)] for i in instances])
    if len(c):
        fig.add_trace(go.Scatter3d(x=c[:, 0], y=c[:, 1], z=c[:, 2], mode="text", text=tx,
                                   textfont=dict(color=INK, size=12), name="ID", showlegend=False))
    return _scene_layout(fig)


def panorama_with_masks(rgb, det_map, detections, station, alpha=0.45):
    """Overlay simulated SAM3 masks (+ boxes, labels) on one panorama -> uint8 image."""
    from PIL import Image, ImageDraw, ImageFont
    img = rgb.astype(np.float32).copy()
    dets = [(k, d) for k, d in enumerate(detections) if d["station"] == station]
    edge = np.zeros(det_map.shape, bool)
    for k, d in dets:
        m = det_map == k
        if not m.any():
            continue
        col = _hex2rgb(CLASS_INFO[d["class"]]["color"]).astype(np.float32)
        img[m] = (1 - alpha) * img[m] + alpha * col
        e = m & ~(np.roll(m, 1, 0) & np.roll(m, -1, 0) & np.roll(m, 1, 1) & np.roll(m, -1, 1))
        img[e] = col
        edge |= e
    im = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).resize((rgb.shape[1] * 2, rgb.shape[0] * 2), Image.NEAREST)
    dr = ImageDraw.Draw(im)
    try:
        font = ImageFont.load_default(size=15)
    except TypeError:
        font = ImageFont.load_default()
    for k, d in dets:
        if d.get("wrap"):
            continue
        r0, c0, r1, c1 = [2 * v for v in d["bbox"]]
        col = tuple(int(v) for v in _hex2rgb(CLASS_INFO[d["class"]]["color"]))
        dr.rectangle([c0, r0, c1, r1], outline=col, width=2)
        label = f"{CLASS_INFO[d['class']]['prompt']} {d['score']:.2f}"
        tw = dr.textlength(label, font=font)
        ty = max(r0 - 19, 0)
        dr.rectangle([c0, ty, c0 + tw + 6, ty + 18], fill=col)
        dr.text((c0 + 3, ty + 1), label, fill=(255, 255, 255), font=font)
    return np.asarray(im)


# =============================================================================== meshes

def _device_template_parts(parts):
    return {p["key"]: p for p in parts if p.get("role") == "device"}


def _rotz90(v, center, turns):
    out = v - center
    for _ in range(turns % 4):
        out = np.stack([-out[:, 1], out[:, 0], out[:, 2]], 1)
    return out + center


def device_mesh(dev, parts_by_key, original_pose):
    """Mesh of a device at its (possibly edited) pose, built from a reconstructed template mesh."""
    ref = dev.get("mesh_ref")
    if ref in parts_by_key and ref in original_pose:
        p = parts_by_key[ref]
        o = original_pose[ref]
        c0 = np.array([o["x"], o["y"], o["z"]])
        turns = int(round((dev.get("yaw", 0) - o.get("yaw", 0)) / 90.0))
        v = _rotz90(p["verts"].astype(np.float64), c0, turns) - c0 + np.array([dev["x"], dev["y"], dev["z"]])
        return v, p["faces"], p["vcolor"]
    lo, hi = device_box(dev)          # fallback: a box
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    f = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    return corners, f, np.full((8, 3), 220, np.uint8)


MESH_GRAY = "#bdbcb7"        # untextured reconstruction
OFF_GRAY = "#9a9994"         # switched-off device / excluded rack


def _mesh_trace(v, f, name, color=None, vcolor=None, opacity=1.0, group=None, showlegend=True):
    # strong directional light so the lumpy reconstructed surface reads through shading alone.
    # (plotly's flatshading=True renders these meshes almost black, so smooth vertex normals are used;
    #  this requires consistently oriented faces, which reconstruction.extract_mesh guarantees)
    kw = dict(x=v[:, 0], y=v[:, 1], z=v[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2], name=name, opacity=opacity,
              flatshading=False, hoverinfo="name", showlegend=showlegend, legendgroup=group or name,
              lighting=dict(ambient=0.32, diffuse=0.85, specular=0.12, roughness=0.6, fresnel=0.08,
                            vertexnormalsepsilon=1e-12, facenormalsepsilon=1e-12),
              lightposition=dict(x=-3000, y=-2000, z=6000))
    if vcolor is not None:
        kw["vertexcolor"] = _rgb(vcolor)
    else:
        kw["color"] = color
    return go.Mesh3d(**kw)


def _part_color(p, color_mode, excluded):
    """gray: everything grey.  thermal: only thermal-model objects coloured (by class), the rest grey."""
    if color_mode == "gray" or p["kind"] == "structure":
        return MESH_GRAY, 1.0
    if p["key"] in excluded:
        return OFF_GRAY, 0.45
    return CLASS_INFO[p["class"]]["color"], 1.0


def mesh_traces(parts, color_mode="gray", hidden=(), devices=None, original_pose=None, only_keys=None, excluded=(),
                role_opacity=None):
    """color_mode: 'gray' (reconstruction only) | 'thermal' (thermal-model objects coloured) | 'rgb' (scan colours)."""
    traces = []
    templates = _device_template_parts(parts)
    for p in parts:
        if p["key"] in hidden or (devices is not None and p.get("role") == "device"):
            continue
        if only_keys is not None and p["kind"] == "structure" and p["key"] not in only_keys:
            continue
        label = p["key"] if p["kind"] == "object" else STRUCTURE_PARTS.get(p["key"], p["key"])
        if color_mode == "rgb":
            traces.append(_mesh_trace(p["verts"], p["faces"], label, vcolor=p["vcolor"]))
        else:
            col, op = _part_color(p, color_mode, excluded)
            if role_opacity and color_mode == "thermal" and p.get("role") in role_opacity:
                op = min(op, role_opacity[p["role"]])
            traces.append(_mesh_trace(p["verts"], p["faces"], label, color=col, opacity=op))
    if devices is not None:
        for d in devices:
            v, f, vc = device_mesh(d, templates, original_pose or {})
            on = d.get("enabled", True)
            name = d["id"] + ("" if on else " (꺼짐)")
            if color_mode == "rgb" and on:
                traces.append(_mesh_trace(v, f, name, vcolor=vc))
            elif color_mode == "gray":
                traces.append(_mesh_trace(v, f, name, color=MESH_GRAY))
            else:
                traces.append(_mesh_trace(v, f, name, color=CLASS_INFO[d["class"]]["color"] if on else OFF_GRAY,
                                          opacity=1.0 if on else 0.35))
    return traces


def light_parts(parts, struct_cap=4000, zone_cap=6000, other_cap=1500):
    """Coarser copy of the mesh parts for views that are redrawn often (editing, slice slider)."""
    import fast_simplification
    out = []
    for p in parts:
        cap = struct_cap if p["kind"] == "structure" else zone_cap if p.get("role") == "zone" else other_cap
        v, f = p["verts"], p["faces"]
        if len(f) > cap * 1.1:
            v2, f2 = fast_simplification.simplify(v.astype(np.float32), f.astype(np.int32), target_reduction=float(1 - cap / len(f)), agg=5)
            if len(f2):
                v, f = v2.astype(np.float32), f2.astype(np.int32)
        q = dict(p)
        q["verts"], q["faces"] = v, f
        q["vcolor"] = np.full((len(v), 3), 190, np.uint8)
        out.append(q)
    return out


def device_label_trace(devices, show_temp=False, min_z=None):
    on = [d for d in devices if d.get("enabled", True)]
    if not on:
        return None
    txt = [f"<b>{d['id']}</b> {d['T_set']:.1f}°C" if show_temp else d["id"] for d in on]
    z = [d["z"] + d["H"] / 2 + 0.45 for d in on]
    if min_z is not None:
        z = [max(v, min_z) for v in z]
    return go.Scatter3d(x=[d["x"] for d in on], y=[d["y"] for d in on], z=z,
                        mode="text", text=txt, textfont=dict(color=INK, size=12), showlegend=False, hoverinfo="skip")


def fig_mesh(parts, color_mode="gray", hidden=(), devices=None, original_pose=None, height=640, excluded=()):
    fig = go.Figure(mesh_traces(parts, color_mode, hidden, devices, original_pose, excluded=excluded))
    if devices:
        t = device_label_trace(devices)
        if t is not None:
            fig.add_trace(t)
    return _scene_layout(fig, height)


# =============================================================================== temperature slices

def slice_plane(T, axis, index):
    if axis == 0:
        return T[index, :, :].T      # rows z, cols y
    if axis == 1:
        return T[:, index, :].T      # rows z, cols x
    return T[:, :, index].T          # rows y, cols x


def plane_axes(axis):
    return {0: (1, 2), 1: (0, 2), 2: (0, 1)}[axis]


def _centers(origin, h, n):
    return origin + (np.arange(n) + 0.5) * h


def _rect_shapes(boxes, axis, coord, color, fill_alpha=0.0, dash=None, width=1.5):
    """Outline of every AABB intersected by the slice plane (axis = coord)."""
    u, v = plane_axes(axis)
    out = []
    for lo, hi in boxes:
        if lo[axis] - 1e-6 <= coord <= hi[axis] + 1e-6:
            out.append(dict(type="rect", x0=lo[u], x1=hi[u], y0=lo[v], y1=hi[v], line=dict(color=color, width=width, dash=dash),
                            fillcolor=color if fill_alpha else None, opacity=1.0 if not fill_alpha else fill_alpha, layer="above"))
    return out


def fig_slice2d(T, origin, h, axis, index, low, high, crange=None, twin=None, devices=None, zone=None,
                height=560, title=None, show_contours=True):
    shape = T.shape
    u, v = plane_axes(axis)
    S = slice_plane(T, axis, index)
    cu = _centers(origin[u], h, shape[u])
    cv = _centers(origin[v], h, shape[v])
    coord = origin[axis] + (index + 0.5) * h
    mid = 0.5 * (low + high)
    zmin, zmax = crange if crange else (mid - 15.0, mid + 15.0)
    fig = go.Figure()
    fig.add_trace(go.Heatmap(x=cu, y=cv, z=S, colorscale=TEMP_COLORSCALE, zmin=zmin, zmax=zmax,
                             colorbar=dict(title=dict(text="°C", font=dict(color=INK)), tickfont=dict(color=INK2), thickness=14),
                             hovertemplate=f"{AXIS_NAMES[u]}=%{{x:.2f}} m<br>{AXIS_NAMES[v]}=%{{y:.2f}} m<br>T=%{{z:.1f}} °C<extra></extra>"))
    if show_contours and np.isfinite(S).any():
        fig.add_trace(go.Contour(x=cu, y=cv, z=np.nan_to_num(S, nan=mid), showscale=False, hoverinfo="skip",
                                 contours=dict(start=low, end=high, size=max(high - low, 1e-3), coloring="none",
                                               showlabels=True, labelfont=dict(color=INK, size=11)),
                                 line=dict(color=INK, width=1.4, dash="dot"), name="목표 범위 경계"))
    shapes, ann = [], []
    if twin is not None:
        racks = [(np.array(r["lo"]), np.array(r["hi"])) for r in twin["racks"] if r.get("in_zone", True)]
        shapes += _rect_shapes(racks, axis, coord, "#1baf7a", width=2)
        hs = [(np.array(x["lo"]), np.array(x["hi"])) for x in twin["heat_sources"]]
        shapes += _rect_shapes(hs, axis, coord, "#eda100", width=1.5)
    if devices is not None:
        for d in devices:
            if not d.get("enabled", True):
                continue
            lo, hi = device_box(d)
            if lo[axis] - h / 2 <= coord <= hi[axis] + h / 2:
                col = CLASS_INFO[d["class"]]["color"]
                shapes.append(dict(type="rect", x0=lo[u], x1=hi[u], y0=lo[v], y1=hi[v], line=dict(color=INK, width=1.5), layer="above"))
                ann.append(dict(x=(lo[u] + hi[u]) / 2, y=hi[v], text=f"{d['id']} {d['T_set']:.0f}°C", showarrow=False, yshift=10,
                                font=dict(size=11, color=INK), bgcolor="rgba(255,255,255,0.85)", bordercolor=col, borderwidth=1))
    title = title or f"{AXIS_NAMES[axis]} = {coord:.2f} m 단면"
    fig.update_layout(
        shapes=shapes, annotations=ann, height=height, margin=dict(l=10, r=10, t=40, b=10),
        title=dict(text=title, font=dict(size=14, color=INK)), plot_bgcolor="#f7f7f5", paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title=AXIS_LABELS[u], color=INK2, showgrid=False, constrain="domain"),
        yaxis=dict(title=AXIS_LABELS[v], color=INK2, showgrid=False, scaleanchor="x", scaleratio=1, constrain="domain"),
        uirevision=f"slice{axis}",
    )
    return fig


def _box_mesh(lo, hi):
    x = [lo[0], hi[0], hi[0], lo[0], lo[0], hi[0], hi[0], lo[0]]
    y = [lo[1], lo[1], hi[1], hi[1], lo[1], lo[1], hi[1], hi[1]]
    z = [lo[2], lo[2], lo[2], lo[2], hi[2], hi[2], hi[2], hi[2]]
    i = [0, 0, 4, 4, 0, 0, 1, 1, 2, 2, 3, 3]
    j = [1, 2, 5, 6, 1, 5, 2, 6, 3, 7, 0, 4]
    k = [2, 3, 6, 7, 5, 4, 6, 5, 7, 6, 4, 7]
    return x, y, z, i, j, k


def _box_edges(lo, hi):
    c = [[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]], [lo[0], lo[1]]]
    xs, ys, zs = [], [], []
    for z in (lo[2], hi[2]):
        for p in c:
            xs.append(p[0]); ys.append(p[1]); zs.append(z)
        xs.append(None); ys.append(None); zs.append(None)
    for p in c[:4]:
        xs += [p[0], p[0], None]; ys += [p[1], p[1], None]; zs += [lo[2], hi[2], None]
    return xs, ys, zs


def context_traces(twin, devices, parts=None, original_pose=None):
    """3D context for the slice view: grey reconstruction (floor + far walls) with only the thermal-model
    objects coloured (racks semi-transparent so the slice stays visible).  Falls back to boxes without parts."""
    out = []
    lo, hi = np.array(twin["domain"]["lo"]), np.array(twin["domain"]["hi"])
    xs, ys, zs = _box_edges(lo, hi)
    out.append(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color=INK2, width=2), hoverinfo="skip", showlegend=False))
    if parts is not None:
        excluded = {r["id"] for r in twin["racks"] if not r.get("in_zone", True)}
        for t in mesh_traces(parts, "thermal", hidden=("ceiling", "clutter"), devices=[d for d in devices or [] if d.get("enabled", True)],
                             original_pose=original_pose, only_keys={"floor", "wall_north", "wall_east"},
                             excluded=excluded, role_opacity={"zone": 0.35, "opening": 0.3}):
            t.showlegend = False
            out.append(t)
        return out
    for r in twin["racks"]:
        x, y, z, i, j, k = _box_mesh(r["lo"], r["hi"])
        out.append(go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color="#1baf7a", opacity=0.18, name=r["id"], hoverinfo="name", showlegend=False, flatshading=True))
        xs, ys, zs = _box_edges(r["lo"], r["hi"])
        out.append(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color="#1baf7a", width=3), hoverinfo="skip", showlegend=False))
    for hs in twin["heat_sources"]:
        x, y, z, i, j, k = _box_mesh(hs["lo"], hs["hi"])
        out.append(go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color="#eda100", opacity=0.9, name=hs["name"], hoverinfo="name", showlegend=False, flatshading=True))
    templates = _device_template_parts(parts) if parts is not None else {}
    for d in devices or []:
        if not d.get("enabled", True):
            continue
        v, f, vc = device_mesh(d, templates, original_pose or {})
        t = _mesh_trace(v, f, d["id"], color=CLASS_INFO[d["class"]]["color"], showlegend=False)
        out.append(t)
    return out


def fig_slice3d(T, origin, h, axis, index, low, high, crange=None, twin=None, devices=None, parts=None, original_pose=None, height=560):
    shape = T.shape
    u, v = plane_axes(axis)
    S = slice_plane(T, axis, index)
    cu = _centers(origin[u], h, shape[u])
    cv = _centers(origin[v], h, shape[v])
    coord = origin[axis] + (index + 0.5) * h
    U, V = np.meshgrid(cu, cv)
    W = np.full_like(U, coord)
    xyz = [None, None, None]
    xyz[u], xyz[v], xyz[axis] = U, V, W
    mid = 0.5 * (low + high)
    zmin, zmax = crange if crange else (mid - 15.0, mid + 15.0)
    fig = go.Figure()
    if twin is not None:
        for t in context_traces(twin, devices, parts, original_pose):
            fig.add_trace(t)
    fig.add_trace(go.Surface(x=xyz[0], y=xyz[1], z=xyz[2], surfacecolor=S, colorscale=TEMP_COLORSCALE, cmin=zmin, cmax=zmax,
                             showscale=False, name="단면", hovertemplate="T=%{surfacecolor:.1f} °C<extra></extra>",
                             lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0)))
    if devices:
        t = device_label_trace(devices, show_temp=True, min_z=coord + 0.6 if axis == 2 else None)
        if t is not None:
            fig.add_trace(t)
    return _scene_layout(fig, height, legend=False)


def fig_zone_hist(T, zone, low, high, height=260):
    z = T[zone]
    z = z[np.isfinite(z)]
    fig = go.Figure()
    fig.add_vrect(x0=low, x1=high, fillcolor="#cde2fb", opacity=0.5, line_width=0, layer="below")
    fig.add_trace(go.Histogram(x=z, xbins=dict(size=0.25), marker=dict(color="#2a78d6", line=dict(color="#fcfcfb", width=1)),
                               hovertemplate="%{x} °C<br>%{y} cells<extra></extra>", name="관리구역 셀"))
    inband = ((z >= low) & (z <= high)).mean() * 100 if len(z) else 0
    fig.add_annotation(x=high, y=1, yref="paper", xanchor="left", yanchor="top", showarrow=False,
                       text=f"목표 범위 내 {inband:.1f}%", font=dict(color=INK, size=12), xshift=6)
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), bargap=0.02, showlegend=False,
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      xaxis=dict(title="관리구역 온도 [°C]", color=INK2, gridcolor=GRID),
                      yaxis=dict(title="셀 수", color=INK2, gridcolor=GRID))
    return fig


# =============================================================================== top view (layout)

def fig_topview(twin, devices, candidates=None, chosen=(), T=None, origin=None, h=None, z_index=None,
                low=15.0, high=25.0, crange=None, height=520, title="평면 배치도"):
    lo, hi = np.array(twin["domain"]["lo"]), np.array(twin["domain"]["hi"])
    fig = go.Figure()
    shapes, ann = [], []
    if T is not None:
        S = slice_plane(T, 2, z_index)
        mid = 0.5 * (low + high)
        zmin, zmax = crange if crange else (mid - 15.0, mid + 15.0)
        fig.add_trace(go.Heatmap(x=_centers(origin[0], h, T.shape[0]), y=_centers(origin[1], h, T.shape[1]), z=S,
                                 colorscale=TEMP_COLORSCALE, zmin=zmin, zmax=zmax, opacity=0.9,
                                 colorbar=dict(title="°C", thickness=12),
                                 hovertemplate="x=%{x:.1f} y=%{y:.1f}<br>T=%{z:.1f} °C<extra></extra>"))
    shapes.append(dict(type="rect", x0=lo[0], x1=hi[0], y0=lo[1], y1=hi[1], line=dict(color=INK, width=2)))
    for r in twin["racks"]:
        inz = r.get("in_zone", True)
        shapes.append(dict(type="rect", x0=r["lo"][0], x1=r["hi"][0], y0=r["lo"][1], y1=r["hi"][1],
                           line=dict(color="#1baf7a", width=2, dash=None if inz else "dot"),
                           fillcolor="#1baf7a" if (inz and T is None) else None, opacity=0.25 if (inz and T is None) else 1.0))
        ann.append(dict(x=(r["lo"][0] + r["hi"][0]) / 2, y=(r["lo"][1] + r["hi"][1]) / 2, showarrow=False,
                        text=r["id"] + (" (관리구역)" if inz else " (제외)"), font=dict(size=11, color=INK)))
    for x in twin["heat_sources"]:
        shapes.append(dict(type="rect", x0=x["lo"][0], x1=x["hi"][0], y0=x["lo"][1], y1=x["hi"][1],
                           line=dict(color="#a87200", width=1.5), fillcolor="#eda100", opacity=0.6))
        ann.append(dict(x=x["lo"][0] if x["lo"][0] > (lo[0] + hi[0]) / 2 else x["hi"][0], y=(x["lo"][1] + x["hi"][1]) / 2,
                        xanchor="right" if x["lo"][0] > (lo[0] + hi[0]) / 2 else "left", showarrow=False, text=x["name"],
                        font=dict(size=10, color=INK), xshift=-8 if x["lo"][0] > (lo[0] + hi[0]) / 2 else 8))
    for d in twin["doors"]:
        w = d["wall"]
        if w in ("wall_south", "wall_north"):
            y = lo[1] if w == "wall_south" else hi[1]
            shapes.append(dict(type="line", x0=d["u0"], x1=d["u1"], y0=y, y1=y, line=dict(color="#6250d6", width=7)))
            ann.append(dict(x=(d["u0"] + d["u1"]) / 2, y=y, text=d["name"], showarrow=False, yshift=-14 if w == "wall_south" else 14, font=dict(size=10, color="#6250d6")))
        else:
            x = lo[0] if w == "wall_west" else hi[0]
            shapes.append(dict(type="line", x0=x, x1=x, y0=d["u0"], y1=d["u1"], line=dict(color="#6250d6", width=7)))
            ann.append(dict(x=x, y=(d["u0"] + d["u1"]) / 2, text=d["name"], showarrow=False, xshift=-28 if w == "wall_west" else 28, font=dict(size=10, color="#6250d6")))
    if candidates:
        cx = [c["x"] for c in candidates]
        cy = [c["y"] for c in candidates]
        chosen = set(chosen or ())
        fig.add_trace(go.Scatter(x=cx, y=cy, mode="markers+text", text=[str(c["cid"]) for c in candidates],
                                 textposition="middle center", textfont=dict(size=8, color=INK),
                                 marker=dict(size=15, symbol="circle", color=["#fab219" if c["cid"] in chosen else "rgba(255,255,255,0.85)" for c in candidates],
                                             line=dict(color=INK2, width=1)),
                                 name="후보 위치", hovertemplate="후보 %{text}<extra></extra>"))
    for d in devices or []:
        dlo, dhi = device_box(d)
        on = d.get("enabled", True)
        col = CLASS_INFO[d["class"]]["color"] if on else "#b5b4ae"
        pad = 0.12        # draw a little bigger so thin devices stay visible
        shapes.append(dict(type="rect", x0=dlo[0] - pad, x1=dhi[0] + pad, y0=dlo[1] - pad, y1=dhi[1] + pad,
                           line=dict(color=INK if on else "#9a9994", width=1.2, dash=None if on else "dot"), fillcolor=col, opacity=1.0 if on else 0.6, layer="above"))
        cxm, cym = (dlo[0] + dhi[0]) / 2, (dlo[1] + dhi[1]) / 2
        # push label into the room
        dx = 0 if d.get("yaw", 0) % 180 == 0 else (1 if cxm < (lo[0] + hi[0]) / 2 else -1)
        dy = (1 if cym < (lo[1] + hi[1]) / 2 else -1) if d.get("yaw", 0) % 180 == 0 else 0
        txt = f"<b>{d['id']}</b>" + (f"<br>{d['T_set']:.1f}°C" if on else "<br>꺼짐")
        ann.append(dict(x=cxm, y=cym, text=txt, showarrow=False, xshift=dx * 34, yshift=dy * 20, font=dict(size=11, color=INK),
                        bgcolor="rgba(255,255,255,0.85)", bordercolor=col, borderwidth=1))
    fig.update_layout(shapes=shapes, annotations=ann, height=height, margin=dict(l=10, r=10, t=36, b=10),
                      title=dict(text=title, font=dict(size=14, color=INK)),
                      plot_bgcolor="#fbfbf9", paper_bgcolor="rgba(0,0,0,0)", showlegend=False,
                      xaxis=dict(range=[lo[0] - 1.2, hi[0] + 1.2], title="x (서→동) [m]", color=INK2, gridcolor=GRID, zeroline=False),
                      yaxis=dict(range=[lo[1] - 1.2, hi[1] + 1.2], title="y (남→북) [m]", color=INK2, gridcolor=GRID, zeroline=False,
                                 scaleanchor="x", scaleratio=1))
    return fig


def fig_voxel_types(cell_type, zone, origin, h, z_index, height=480):
    """Categorical map of voxel roles at one height."""
    S = cell_type[:, :, z_index].T.astype(float)
    Z = zone[:, :, z_index].T
    S = np.where((S == 1) & Z, 5, S)       # goods inside managed zone
    names = ["공기", "화물(랙)", "장치(ID)", "발열설비(BC)", "단열(제외)", "관리구역"]
    cols = ["#f7f7f5", "#9ec5f4", "#eb6834", "#eda100", "#52514e", "#1baf7a"]
    cs = []
    for i, c in enumerate(cols):
        cs += [[i / 6, c], [(i + 1) / 6, c]]
    fig = go.Figure(go.Heatmap(x=_centers(origin[0], h, cell_type.shape[0]), y=_centers(origin[1], h, cell_type.shape[1]),
                               z=S, zmin=-0.5, zmax=5.5, colorscale=cs, xgap=0.5, ygap=0.5,
                               colorbar=dict(tickvals=list(range(6)), ticktext=names, thickness=12),
                               customdata=np.array(names, dtype=object)[S.astype(int)],
                               hovertemplate="x=%{x:.2f} y=%{y:.2f}<br>%{customdata}<extra></extra>"))
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="rgba(0,0,0,0)",
                      xaxis=dict(title="x [m]", color=INK2, showgrid=False), yaxis=dict(title="y [m]", color=INK2, showgrid=False, scaleanchor="x"))
    return fig
