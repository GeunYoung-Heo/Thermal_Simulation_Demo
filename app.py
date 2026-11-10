"""Thermal Digital Twin demo - Streamlit app.

    streamlit run app.py
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from thermal_twin import io, optimize, slice_component, thermal, viz  # noqa: E402
from thermal_twin import twin as twinmod  # noqa: E402
from thermal_twin.common import CLASS_INFO, CLASS_NAMES, STRUCTURE_PARTS  # noqa: E402
from thermal_twin.scene import load_config  # noqa: E402

DATA = ROOT / "data"
RESULTS = ROOT / "results"
CONFIG = ROOT / "configs" / "warehouse.yaml"

st.set_page_config(page_title="Thermal Digital Twin", page_icon="🌡️", layout="wide")

PAGES = ["⓪ 개요", "① 3D 스캔", "② SAM3 탐지·재매핑", "③ 메쉬 재구성", "④ 열 모델 설정",
         "⑤ 열 시뮬레이션", "⑥ 배치·운전 최적화", "⑦ 결과 뷰어"]
ROLE_LABEL = {"device": "제어 장치 → ID·설계변수", "heat_source": "발열설비 → 경계조건",
              "zone": "보관 랙 → 관리대상 구역", "opening": "출입문 → 경계조건"}
BC_TYPE_HELP = "convective: 외기온도 T와 열관류율 U로 열 출입 · fixed: 표면온도를 T로 고정 · adiabatic: 단열(열 출입 없음)"
DEV_COLS = ["id", "class", "enabled", "x", "y", "z", "yaw", "T_set", "L", "T", "H", "mesh_ref", "wall"]
BC_COLS = ["name", "label", "kind", "type", "T", "U"]
ss = st.session_state


# =============================================================================== data

def data_ready():
    need = ["scan.npz", "segmentation.npz", "segmentation.json", "mesh_display.npz", "mesh_parts.json", "twin.json"]
    return all((DATA / f).exists() for f in need)


@st.cache_resource
def get_cfg():
    return load_config(CONFIG)


@st.cache_resource
def get_scan():
    return io.load_scan(DATA)


@st.cache_resource
def get_seg():
    return io.load_segmentation(DATA)


@st.cache_resource
def get_mesh():
    return io.load_mesh(DATA)


@st.cache_resource
def get_twin():
    return io.load_twin(DATA)


@st.cache_resource
def get_light_mesh():
    """coarser mesh for views that are redrawn on every edit"""
    return viz.light_parts(get_mesh()[0])


@st.cache_resource
def get_result(path: str, mtime: float):
    return io.load_result(path)


def original_pose():
    return {d["mesh_ref"]: d for d in get_twin()["devices"]}


# =============================================================================== state helpers

def devices_df(devs):
    return pd.DataFrame([{c: d.get(c) for c in DEV_COLS} for d in devs], columns=DEV_COLS)


def bcs_df(bcs):
    return pd.DataFrame([{c: b.get(c) for c in BC_COLS} for b in bcs], columns=BC_COLS)


def _missing(v):
    return v is None or (isinstance(v, float) and np.isnan(v)) or (isinstance(v, str) and not v.strip()) or (not isinstance(v, (str, bool)) and pd.isna(v))


def df_to_devices(df):
    cfg, tw = get_cfg(), get_twin()
    lo, hi = np.array(tw["domain"]["lo"]), np.array(tw["domain"]["hi"])
    out, ids = [], set()
    for r in df.to_dict("records"):
        cls = r.get("class") if r.get("class") in cfg["thermal"]["device_types"] else "radiator"
        tpl = twinmod.new_device(tw, cfg, cls, ids | {x for x in df["id"].tolist() if isinstance(x, str)})
        d = {}
        for c in DEV_COLS:
            v = r.get(c)
            d[c] = tpl.get(c) if _missing(v) else v
        d["class"] = cls
        if d["id"] in ids:
            d["id"] = twinmod.new_device(tw, cfg, cls, ids)["id"]
        ids.add(d["id"])
        d["enabled"] = bool(d["enabled"])
        d["yaw"] = int(d["yaw"]) if int(d["yaw"]) in (0, 90) else 0
        for k, a in (("x", 0), ("y", 1), ("z", 2)):
            d[k] = float(np.clip(float(d[k]), lo[a], hi[a]))
        for k in ("T_set", "L", "T", "H"):
            d[k] = float(d[k])
        out.append(d)
    return out


def df_to_bcs(df):
    out = []
    for r in df.to_dict("records"):
        b = {c: r.get(c) for c in BC_COLS}
        if b["type"] not in ("convective", "fixed", "adiabatic"):
            b["type"] = "convective"
        b["T"] = 0.0 if _missing(b["T"]) else float(b["T"])
        b["U"] = None if _missing(b["U"]) else float(b["U"])
        if b["type"] == "convective" and b["U"] is None:
            b["type"] = "fixed"
        out.append(b)
    return out


def set_devices(devs):
    ss.devices = devs
    ss.dev_base = devices_df(devs)
    ss.dev_ver = ss.get("dev_ver", 0) + 1


def set_bcs(bcs):
    ss.bcs = bcs
    ss.bc_base = bcs_df(bcs)
    ss.bc_ver = ss.get("bc_ver", 0) + 1


def init_state(force=False):
    cfg, tw = get_cfg(), get_twin()
    if force or "devices" not in ss:
        preset = ss.get("preset_sel", "winter")
        set_devices(twinmod.preset_devices(tw, cfg, preset))
        set_bcs(twinmod.default_boundaries(tw, cfg, preset))
        ss.zone_racks = [r["id"] for r in tw["racks"]]
        ss.sim = None
        ss.sim_key = None
        ss.opt = None


def apply_preset():
    cfg, tw = get_cfg(), get_twin()
    preset = ss.preset_sel
    active = set(cfg["thermal"]["presets"][preset]["active_device_classes"])
    devs = copy.deepcopy(ss.devices)
    for d in devs:
        d["enabled"] = d["class"] in active
    set_devices(devs)
    set_bcs(twinmod.default_boundaries(tw, cfg, preset))
    th = cfg["thermal"]["target"]
    ss.band = (th["low"], th["high"])


def twin_eff():
    tw = copy.deepcopy(get_twin())
    for r in tw["racks"]:
        r["in_zone"] = r["id"] in ss.zone_racks
    return tw


def settings():
    return ss.h, ss.k_air, ss.k_goods, ss.band[0], ss.band[1], ss.margin


def model_key():
    h, ka, kg, lo, hi, m = settings()
    blob = json.dumps([ss.devices, ss.bcs, sorted(ss.zone_racks), h, ka, kg, lo, hi], sort_keys=True, default=str)
    return hashlib.md5(blob.encode()).hexdigest()


def device_power_kw(res):
    bh = res.meta.get("body_heat_W", {})
    return sum(abs(bh.get(n, 0.0)) for n, k in zip(res.grid.body_names, res.grid.body_kind) if k == "device") / 1000


def envelope_loss_kw(res):
    return -sum(res.meta.get("group_heat_W", {}).values()) / 1000


def run_pipeline():
    spec = importlib.util.spec_from_file_location("build_twin", ROOT / "scripts" / "build_twin.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    with st.status("파이프라인 실행 중 (약 30초)...", expanded=True) as s:
        mod.run(CONFIG, DATA, log=st.write)
        s.update(label="파이프라인 완료", state="complete")
    st.cache_resource.clear()
    for k in ("devices", "sim", "opt"):
        ss.pop(k, None)


# =============================================================================== shared widgets

def plot(fig, key=None):
    st.plotly_chart(fig, width="stretch", theme=None, key=key, config={"displaylogo": False})


def slice_viewer(T, origin, h, zone, devices, key, low, high):
    """Section viewer (custom component): the slider runs in the browser, so the 2D heatmap follows the
    slider while dragging and nothing is re-sent / re-drawn by Streamlit (no flicker)."""
    slice_component.slice_viewer(T, origin, h, low, high, twin_eff(), devices, get_light_mesh(), original_pose(), key=key)
    st.caption("점선 = 목표 온도 범위 경계 · 초록 테두리 = 관리대상 구역(랙) · 노란 테두리 = 발열설비(경계조건) · 검은 테두리 = ID 장치 · "
               "2D 단면은 드래그하는 동안 바로 따라오고, 3D 단면은 슬라이더를 멈추거나 놓으면 갱신됩니다. 키보드 ←/→로 한 칸씩 움직일 수도 있습니다.")


def result_metrics(res, low, high, cols=None):
    m = res.meta
    cols = cols or st.columns(6)
    cols[0].metric("목표 범위 만족률", f"{100 * m.get('zone_in_band', 0):.1f} %")
    cols[1].metric("관리구역 최저", f"{m.get('zone_min', float('nan')):.1f} °C")
    cols[2].metric("관리구역 평균", f"{m.get('zone_mean', float('nan')):.1f} °C")
    cols[3].metric("관리구역 최고", f"{m.get('zone_max', float('nan')):.1f} °C")
    cols[4].metric("장치 열량 합계", f"{device_power_kw(res):.2f} kW")
    cols[5].metric("외피 열 출입", f"{-envelope_loss_kw(res):+.2f} kW", help="벽·지붕·바닥·문을 통해 들어오는(+) / 나가는(-) 열")


def heat_tables(res):
    labels = {b["name"]: b.get("label", b["name"]) for b in res.bcs}
    rows = []
    for n, k in zip(res.grid.body_names, res.grid.body_kind):
        T = next((d["T_set"] for d in res.devices if d["id"] == n), None) if k == "device" else next((b["T"] for b in res.bcs if b["name"] == n), None)
        rows.append({"이름": n if k == "device" else labels.get(n, n), "구분": "ID 장치" if k == "device" else "발열설비(BC)",
                     "표면온도 [°C]": T, "공급 열량 [kW]": res.meta["body_heat_W"].get(n, 0) / 1000})
    g = [{"경계": labels.get(n, n), "열 출입 [kW]": q / 1000} for n, q in res.meta.get("group_heat_W", {}).items()]
    return pd.DataFrame(rows), pd.DataFrame(g)


# =============================================================================== sidebar

if not data_ready():
    st.title("🌡️ Thermal Digital Twin")
    st.warning("재구성 데이터(data/)가 없습니다. 먼저 파이프라인을 실행하세요. (약 30초)")
    if st.button("파이프라인 실행", type="primary"):
        run_pipeline()
        st.rerun()
    st.stop()

cfg = get_cfg()
th_cfg = cfg["thermal"]
init_state()
# widget defaults live in session_state (so presets / callbacks can change them without warnings)
ss.setdefault("band", (float(th_cfg["target"]["low"]), float(th_cfg["target"]["high"])))
ss.setdefault("margin", float(th_cfg["target"]["margin"]))
ss.setdefault("h", float(th_cfg["grid_h"]))
ss.setdefault("k_air", float(th_cfg["materials"]["air"]))
ss.setdefault("k_goods", float(th_cfg["materials"]["goods"]))

with st.sidebar:
    st.markdown("### 🌡️ Thermal Digital Twin")
    st.caption(f"{cfg['name']} · 연구 시스템 시연용 데모")
    page = st.radio("단계", PAGES, key="page", label_visibility="collapsed")
    st.divider()
    presets = th_cfg["presets"]
    st.selectbox("시나리오", list(presets), format_func=lambda k: presets[k]["label"], key="preset_sel", on_change=apply_preset,
                 help="외기 조건과 사용하는 장치 종류를 한 번에 바꿉니다. (경계조건 표가 초기화됩니다)")
    st.slider("관리구역 목표 온도 [°C]", 0.0, 40.0, step=0.5, key="band")
    st.number_input("안전 여유 [°C]", 0.0, 3.0, step=0.1, key="margin",
                    help="최적화 시 목표 범위 안쪽으로 이만큼 여유를 두고 맞춥니다.")
    with st.expander("해석 설정"):
        st.select_slider("voxel 크기 [m]", options=[0.5, 0.4, 0.33, 0.25, 0.2], key="h",
                         help="작을수록 정밀하지만 느립니다. 0.25 m = 약 19만 셀")
        st.number_input("공기 유효 열전도도 [W/m·K]", 0.5, 200.0, step=1.0, key="k_air",
                        help="실제 공기는 0.026 W/m·K이지만, 전도만 고려하므로 대류 혼합 효과를 등가 전도로 반영한 데모용 값을 씁니다.")
        st.number_input("화물(랙) 유효 열전도도 [W/m·K]", 0.1, 200.0, step=0.5, key="k_goods")
    with st.expander("파이프라인 재실행"):
        st.caption("configs/warehouse.yaml을 수정한 뒤 스캔부터 다시 만듭니다. 편집 중인 장치/경계조건은 초기화됩니다.")
        if st.button("스캔 → 탐지 → 메쉬 → 트윈 다시 만들기"):
            run_pipeline()
            st.rerun()

# page change: re-sync the editors' base tables with the current state
if ss.get("last_page") != page:
    if page == PAGES[4]:
        ss.dev_base = devices_df(ss.devices)
        ss.bc_base = bcs_df(ss.bcs)
        ss.dev_ver = ss.get("dev_ver", 0) + 1
        ss.bc_ver = ss.get("bc_ver", 0) + 1
        ss.zone_w = list(ss.zone_racks)
    ss.last_page = page

low, high = ss.band
margin = ss.margin


# =============================================================================== ⓪ overview

if page == PAGES[0]:
    scan, seg = get_scan(), get_seg()
    parts, mstats = get_mesh()
    tw = get_twin()
    st.title("실내 열관리 디지털 트윈 — 의약품 물류창고")
    st.markdown(
        f"의약품은 GDP(Good Distribution Practice) 규정에 따라 **{low:.0f}–{high:.0f} °C**(실온 보관 기준)에서 보관해야 하며, "
        "창고는 정기적으로 **온도 매핑**을 해야 합니다. 이 데모는 공간을 스캔해 만든 디지털 트윈에서 다음 질문에 답합니다.")
    st.info(f"**관리대상 구역(보관 랙)의 온도를 {low:.0f} °C에서 {high:.0f} °C 사이로 유지하려면, "
            "열 관련 기구를 어디에 배치하고 몇 도로 운용해야 하는가?**")
    steps = [
        ("① 3D 스캔", f"{len(scan.positions)}곳 360° RGB-D 스캔", f"{len(scan.points):,} 포인트"),
        ("② SAM3 탐지·재매핑", "이미지 마스크 → 포인트 투표 → 3D 객체", f"{len(seg.instances)}개 객체 · IoU {seg.stats['mean_iou']:.2f}"),
        ("③ 메쉬 재구성", "TSDF 융합 + 마칭 큐브, 객체별 ID", f"{mstats['full_faces']:,} 삼각형"),
        ("④ 열 모델", "ID 장치 = 설계변수 / 나머지 = 경계조건", f"장치 {len(tw['devices'])} · BC {len(tw['heat_sources']) + len(tw['doors']) + 6}"),
        ("⑤ 열 시뮬레이션", "voxel 정상상태 전도 해석", "약 19만 셀 · ~1초"),
        ("⑥ 최적화", "배치 탐색 + 운전온도 LP", "어디에 · 몇 도로"),
    ]
    cols = st.columns(len(steps))
    for c, (a, b, n) in zip(cols, steps):
        with c.container(border=True):
            st.markdown(f"**{a}**")
            st.caption(b)
            st.markdown(f"`{n}`")
    st.image(viz.panorama_with_masks(scan.rgb[5], get_seg().det_map[5], seg.detections, 5),
             caption="스캔 스테이션 S6의 360° 파노라마 위에 SAM3(시뮬레이션) 탐지 결과를 겹친 모습", width="stretch")
    with st.expander("이 데모에서 '시뮬레이션'으로 대체한 부분과 단순화 가정"):
        st.markdown(
            "- **현장**: 실제 창고 대신 `configs/warehouse.yaml`로 정의한 가상 창고를 사용합니다. 이 정답은 스캐너/SAM3 흉내에만 쓰이고, "
            "이후 단계는 재구성 결과만 사용합니다.\n"
            "- **스캔**: 레이캐스팅으로 360° RGB-D 파노라마를 만들고 거리 비례 노이즈, 누락, flying pixel, 스테이션 정합 오차를 넣었습니다.\n"
            "- **SAM3**: 정답 라벨에서 마스크를 만든 뒤 경계 노이즈, 팽창/침식, 작은 객체 미탐지, 신뢰도 점수를 넣었습니다. "
            "실제 SAM3 출력(마스크+클래스+점수)으로 교체하면 이후 단계는 그대로 동작합니다.\n"
            "- **열해석**: 전도만 고려합니다. 공기의 대류 혼합 효과는 '유효 열전도도'로 등가 반영한 값이며 공학적 정확도보다 시스템 동작을 보여주는 것이 목적입니다.")


# =============================================================================== ① scan

elif page == PAGES[1]:
    scan, seg = get_scan(), get_seg()
    st.header("① 3D 스캔: 공간을 포인트클라우드로")
    st.caption("창고 안 12곳에서 360° RGB-D 파노라마를 촬영(가상)하고 정합해 하나의 포인트클라우드로 합칩니다. "
               "거리에 따른 노이즈, 가려진 곳의 구멍(랙 윗면 등), 경계의 flying pixel, 스테이션 간 정합 오차가 일부러 포함되어 있습니다.")
    c = st.columns(4)
    c[0].metric("스캔 위치", f"{len(scan.positions)} 곳")
    c[1].metric("포인트 수", f"{len(scan.points):,}")
    c[2].metric("이상점(제거 대상)", f"{int((~seg.inlier).sum()):,}")
    c[3].metric("파노라마 해상도", f"{scan.geom.width}×{scan.geom.height}")
    left, right = st.columns([1.6, 1])
    with right:
        hide_ceiling = st.toggle("천장 숨기기", True)
        show_out = st.toggle("이상점(flying pixel) 빨갛게 표시", False)
        npts = st.slider("표시 포인트 수", 50_000, 300_000, 150_000, 25_000)
        s = st.selectbox("파노라마 보기", range(len(scan.positions)), format_func=lambda i: f"스테이션 S{i + 1}", index=5)
        st.image(scan.rgb[s], caption=f"S{s + 1} 360° 파노라마 (RGB)", width="stretch")
    with left:
        zmax = get_seg().room["z1"] - 0.6 if hide_ceiling else None
        plot(viz.fig_pointcloud(scan.points, scan.colors, npts, zmax, scan.positions, (~seg.inlier) if show_out else None))


# =============================================================================== ② SAM3

elif page == PAGES[2]:
    scan, seg = get_scan(), get_seg()
    st.header("② SAM3 탐지 → 포인트클라우드 재매핑")
    st.caption("텍스트 프롬프트: " + " · ".join(f"`{CLASS_INFO[c]['prompt']}`" for c in CLASS_NAMES) +
               "  →  각 파노라마에서 마스크 생성 → 3D 포인트를 모든 이미지에 투영해 마스크 클래스를 투표(가까운 시점 가중) "
               "→ 같은 클래스 포인트를 3D로 클러스터링 → 객체마다 ID 부여")
    c = st.columns(4)
    c[0].metric("2D 마스크 수", seg.stats["n_detections_2d"])
    c[1].metric("3D 객체 수", len(seg.instances))
    c[2].metric("평균 IoU (정답 대비)", f"{seg.stats['mean_iou']:.3f}", help="가상 현장의 정답 라벨과 비교한 검증용 수치")
    c[3].metric("라벨이 붙은 포인트", f"{100 * seg.stats['labelled_ratio']:.1f} %")
    s = st.select_slider("스테이션", options=list(range(len(scan.positions))), value=5, format_func=lambda i: f"S{i + 1}")
    st.image(viz.panorama_with_masks(scan.rgb[s], seg.det_map[s], seg.detections, s),
             caption=f"S{s + 1}: SAM3(시뮬레이션) 마스크 · 클래스 · 신뢰도", width="stretch")
    left, right = st.columns([1.5, 1])
    with left:
        hide_ceiling = st.toggle("천장 숨기기", True, key="seg_hide")
        zmax = seg.room["z1"] - 0.6 if hide_ceiling else None
        plot(viz.fig_labeled_pointcloud(scan.points, seg.point_instance, seg.point_struct, seg.instances, 160_000, zmax))
    with right:
        st.markdown("**탐지된 객체**")
        df = pd.DataFrame([{
            "ID": i["id"], "종류": CLASS_INFO[i["class"]]["label"], "역할": ROLE_LABEL[i["role"]],
            "포인트": i["n_points"], "신뢰도": i["confidence"], "IoU": i.get("iou"),
        } for i in seg.instances])
        st.dataframe(df, hide_index=True, height=560,
                     column_config={"신뢰도": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f"),
                                    "IoU": st.column_config.NumberColumn(format="%.2f")})
        st.caption("ID가 붙지 않은 포인트는 평면 검출로 바닥·벽·천장(→ 경계조건)과 기타로 나뉩니다.")


# =============================================================================== ③ mesh

elif page == PAGES[3]:
    parts, mstats = get_mesh()
    st.header("③ 메쉬 재구성")
    st.caption("12개 깊이 파노라마를 TSDF(Truncated Signed Distance Field, 5 cm voxel)로 융합하고 마칭 큐브로 표면을 추출했습니다. "
               "각 삼각형은 가장 가까운 라벨 포인트의 ID를 물려받아 객체별 메쉬로 나뉩니다. 울퉁불퉁한 면, 구멍, 살짝 겹친 벽은 실제 스캔 재구성의 전형적인 모습입니다.")
    c = st.columns(4)
    c[0].metric("전체 삼각형", f"{mstats['full_faces']:,}")
    c[1].metric("표시용 삼각형", f"{mstats['display_faces']:,}")
    c[2].metric("TSDF voxel", f"{mstats['tsdf_voxel'] * 100:.0f} cm")
    c[3].metric("메쉬 파트", f"{len(parts)} 개")
    keys = [p["key"] for p in parts]
    names = {p["key"]: (p["key"] if p["kind"] == "object" else f"{STRUCTURE_PARTS.get(p['key'], p['key'])} ({p['key']})") for p in parts}
    hidden = st.multiselect("숨길 파트 — 내부가 보이도록 카메라 쪽(남·서) 벽과 천장을 기본으로 숨겼습니다. 지우면 다시 나타납니다.",
                            keys, default=[k for k in ("ceiling", "wall_south", "wall_west") if k in keys], format_func=lambda k: names[k])
    plot(viz.fig_mesh(parts, "gray", set(hidden), height=700))
    st.caption(f"전체 해상도 메쉬(파트 번호 포함): `{(DATA / 'mesh_full.ply').relative_to(ROOT)}` — MeshLab/CloudCompare에서 열 수 있습니다. "
               "범례를 클릭해도 파트를 켜고 끌 수 있습니다. 북측 벽은 0.3 m 앞에 선 랙(RACK-03)에 가려 스캐너가 볼 수 없는 부분이 비어 있습니다.")


# =============================================================================== ④ thermal model

elif page == PAGES[4]:
    tw0 = get_twin()
    st.header("④ 열 모델 설정")
    st.caption("재구성 결과에서 자동 생성된 열 모델입니다. **ID가 붙은 장치**는 위치·온도를 바꿀 수 있는 설계 변수이고, "
               "**ID가 없는 벽·문·발열설비**는 경계조건입니다. 표를 직접 편집하세요. (행 추가/삭제 가능)")
    left, right = st.columns([1.05, 1])
    with left:
        st.subheader("장치 (ID 부여)")
        dev_types = list(th_cfg["device_types"])
        edited = st.data_editor(
            ss.dev_base, key=f"dev_editor_{ss.dev_ver}", num_rows="dynamic", hide_index=True, width="stretch",
            column_config={
                "id": st.column_config.TextColumn("ID", help="비워두면 자동 부여"),
                "class": st.column_config.SelectboxColumn("종류", options=dev_types, required=True, default="radiator"),
                "enabled": st.column_config.CheckboxColumn("사용", default=True),
                "x": st.column_config.NumberColumn("x [m]", format="%.2f", step=0.1),
                "y": st.column_config.NumberColumn("y [m]", format="%.2f", step=0.1),
                "z": st.column_config.NumberColumn("z [m]", format="%.2f", step=0.05),
                "yaw": st.column_config.SelectboxColumn("방향 [°]", options=[0, 90], help="0 = x축과 나란히, 90 = y축과 나란히"),
                "T_set": st.column_config.NumberColumn("설정온도 [°C]", format="%.1f", step=0.5),
                "L": None, "T": None, "H": None, "mesh_ref": None, "wall": None,
            })
        ss.devices = df_to_devices(edited)
        b1, b2, b3 = st.columns(3)
        if b1.button("벽에 붙이기", help="각 장치를 가장 가까운 벽에 붙이고 방향을 맞춥니다.", width="stretch"):
            devs = []
            for d in ss.devices:
                wall = twinmod.nearest_wall((d["x"], d["y"]), {"x0": tw0["domain"]["lo"][0], "x1": tw0["domain"]["hi"][0],
                                                               "y0": tw0["domain"]["lo"][1], "y1": tw0["domain"]["hi"][1]})
                u = d["x"] if wall in ("wall_south", "wall_north") else d["y"]
                nd = twinmod.new_device(tw0, cfg, d["class"], set(), wall, u)
                devs.append({**d, **{k: nd[k] for k in ("x", "y", "z", "yaw", "wall")}})
            set_devices(devs)
            st.rerun()
        if b2.button("탐지된 배치로 되돌리기", width="stretch"):
            set_devices(twinmod.preset_devices(tw0, cfg, ss.preset_sel))
            st.rerun()
        if b3.button("라디에이터 추가", width="stretch", help="북측 벽 가운데에 새 라디에이터를 추가합니다."):
            nd = twinmod.new_device(tw0, cfg, "radiator", {d["id"] for d in ss.devices}, "wall_north")
            set_devices(ss.devices + [nd])
            st.rerun()

        st.subheader("경계조건 (ID 없음)")
        bc_edit = st.data_editor(
            ss.bc_base, key=f"bc_editor_{ss.bc_ver}", num_rows="fixed", hide_index=True, width="stretch",
            disabled=["name", "label", "kind"],
            column_config={
                "name": None,
                "label": st.column_config.TextColumn("경계"),
                "kind": st.column_config.TextColumn("구분"),
                "type": st.column_config.SelectboxColumn("유형", options=["convective", "fixed", "adiabatic"], help=BC_TYPE_HELP),
                "T": st.column_config.NumberColumn("온도 T [°C]", format="%.1f", step=0.5),
                "U": st.column_config.NumberColumn("열관류율 U [W/m²K]", format="%.2f", step=0.05),
            })
        ss.bcs = df_to_bcs(bc_edit)
        st.caption(BC_TYPE_HELP + " · 발열설비는 fixed(가동) 또는 adiabatic(정지)만 의미가 있습니다.")

        st.subheader("관리대상 구역")
        ss.setdefault("zone_w", list(ss.zone_racks))
        ss.zone_racks = st.multiselect("온도를 관리할 랙", [r["id"] for r in tw0["racks"]], key="zone_w",
                                       help="SAM3로 탐지된 보관 랙이 관리대상 구역이 됩니다.")
    with right:
        tw = twin_eff()
        plot(viz.fig_topview(tw, ss.devices, height=560, title="평면 배치도 (편집 내용 실시간 반영)"))
        with st.expander("3D로 보기 — 열 모델 요소만 색으로 표시 (편집 내용 반영)", expanded=True):
            excluded = {r["id"] for r in tw["racks"] if not r.get("in_zone", True)}
            plot(viz.fig_mesh(get_light_mesh(), "thermal", {"ceiling", "wall_south", "wall_west"}, ss.devices, original_pose(),
                              height=520, excluded=excluded))
            st.caption("회색 = 재구성 메쉬(구조물·기타) · 주황 = 라디에이터 · 파랑 = 냉방기 · 노랑/분홍 = 발열설비 · 초록 = 관리대상 랙 · 보라 = 출입문 · 반투명 회색 = 꺼짐/제외")
        with st.expander("voxel 모델 미리보기"):
            h = ss.h
            g = thermal.build_grid(tw, ss.devices, ss.bcs, h, ss.k_air, ss.k_goods)
            zc = g.origin[2] + (np.arange(g.shape[2]) + 0.5) * h
            zsel = st.slider("높이 [m]", float(zc[0]), float(zc[-1]), float(zc[np.argmin(np.abs(zc - 0.45))]), float(h), key="vox_z")
            plot(viz.fig_voxel_types(g.cell_type, g.zone, g.origin, h, int(np.argmin(np.abs(zc - zsel))), height=420))
            st.caption(f"격자 {g.shape[0]}×{g.shape[1]}×{g.shape[2]} = {np.prod(g.shape):,} 셀 (voxel {h} m)")


# =============================================================================== ⑤ simulation

elif page == PAGES[5]:
    st.header("⑤ 열 시뮬레이션")
    st.caption("공간을 voxel로 나누고 정상상태 열전도 방정식 ∇·(k∇T)=0 을 풉니다. ID 장치 = 설정 표면온도, 외벽/지붕/바닥/문 = 외기온도+열관류율, "
               "발열설비 = 고정 온도. ④에서 편집한 내용이 그대로 반영됩니다.")
    run = st.button("▶ 시뮬레이션 실행", type="primary")
    if run:
        h, ka, kg, lo_, hi_, _ = settings()
        with st.spinner("행렬 조립 및 AMG 풀이 중..."):
            ss.sim = thermal.simulate(twin_eff(), ss.devices, ss.bcs, h, ka, kg, (lo_, hi_), label="사용자 설정")
            ss.sim_key = model_key()
    res = ss.sim
    if res is None:
        st.info("▶ 버튼을 눌러 현재 설정으로 시뮬레이션하세요.")
    else:
        if ss.sim_key != model_key():
            st.warning("설정이 바뀌었습니다. 아래 결과는 이전 설정 기준입니다. 다시 실행하세요.")
        result_metrics(res, low, high)
        slice_viewer(res.T, res.grid.origin, res.grid.h, res.grid.zone, res.devices, "sim", low, high)
        a, b = st.columns([1, 1])
        with a:
            st.markdown("**관리구역 온도 분포**")
            plot(viz.fig_zone_hist(res.T, res.grid.zone, low, high))
        with b:
            dfb, dfg = heat_tables(res)
            st.markdown("**열원별 공급 열량**")
            st.dataframe(dfb, hide_index=True, column_config={"공급 열량 [kW]": st.column_config.NumberColumn(format="%.2f"),
                                                              "표면온도 [°C]": st.column_config.NumberColumn(format="%.1f")})
            st.markdown("**경계별 열 출입** (음수 = 손실)")
            st.dataframe(dfg, hide_index=True, column_config={"열 출입 [kW]": st.column_config.NumberColumn(format="%.2f")})
        with st.form("save_sim"):
            c1, c2 = st.columns([3, 1])
            name = c1.text_input("저장 이름", value=f"{ss.preset_sel}_사용자설정")
            if c2.form_submit_button("💾 결과 저장", width="stretch"):
                p = io.save_result(res, RESULTS, name)
                st.success(f"저장됨: {p.relative_to(ROOT)}  →  ⑦ 결과 뷰어에서 다시 볼 수 있습니다.")
        st.caption(f"격자 {res.meta['grid']['shape']} · 미지수 {res.meta['grid']['unknowns']:,} · 계산 {res.meta['solve_seconds']} s")


# =============================================================================== ⑥ optimisation

elif page == PAGES[6]:
    tw = twin_eff()
    st.header("⑥ 배치·운전 최적화")
    st.info(f"**질문:** 관리대상 구역(랙)의 모든 셀을 **{low:.1f}–{high:.1f} °C** (안전 여유 {margin:.1f} °C) 안에 두려면, "
            "열 관련 기구를 **어디에** 두고 **몇 도로** 운용해야 하는가?")
    with st.expander("어떻게 푸나요?"):
        st.markdown(
            "1. **운전온도 (배치 고정)**: 전도 문제는 선형이므로 온도장은 `T(x) = T_기준(x) + Σ θᵢ·Gᵢ(x)` 로 분해됩니다. "
            "(θᵢ = 장치 i의 표면온도, Gᵢ = 단위 응답). 장치 수+1번만 풀면 되고, 이후는 **선형계획(LP)** 입니다.\n"
            "   - 1단계: 범위 위반량 `s`를 최소화 (s ≤ 0 이면 모든 관리구역 셀이 범위 안)\n"
            "   - 2단계: 범위를 지키는 해 중 **장치 열량 합(에너지)을 최소화**\n"
            "2. **배치**: 벽을 따라 후보 위치를 만들고, 각 후보의 단위 응답을 한 번씩 미리 계산합니다. "
            "그러면 어떤 조합이든 작은 LP 한 번으로 평가할 수 있어 **탐욕 추가 + 교환 탐색**으로 좋은 조합을 찾습니다. "
            "마지막에 고른 위치에 장치를 실제로 넣고 1번 방법으로 정확한 운전온도를 다시 계산합니다.")
    mode = st.radio("최적화 범위", ["현재 배치 유지 → 운전온도만 최적화", "배치 + 운전온도 동시 최적화"], horizontal=True, index=1)
    cands, opt_cls = None, None
    if mode.startswith("배치"):
        c1, c2, c3, c4 = st.columns([1, 1, 1, 1.4])
        active = th_cfg["presets"][ss.preset_sel]["active_device_classes"]
        dev_types = list(th_cfg["device_types"])
        opt_cls = c1.selectbox("배치할 장치 종류", dev_types, index=dev_types.index(active[0]), format_func=lambda k: CLASS_INFO[k]["label"])
        n_now = sum(1 for d in ss.devices if d["class"] == opt_cls and d.get("enabled", True))
        n_dev = c2.number_input("장치 수", 1, 10, max(n_now, 1))
        spacing = c3.slider("후보 간격 [m]", 1.0, 4.0, 2.0, 0.5)
        walls = c4.multiselect("설치 가능한 벽", optimize.WALLS, default=optimize.WALLS,
                               format_func=lambda w: {"wall_south": "남", "wall_north": "북", "wall_west": "서", "wall_east": "동"}[w])
        objective = st.radio("배치 선택 기준", ["margin", "energy"], horizontal=True,
                             format_func=lambda k: {"margin": "여유 최대 (온도 균일성 우선)", "energy": "에너지 최소"}[k])
        keep = [d for d in ss.devices if d["class"] != opt_cls and d.get("enabled", True)]
        cands = optimize.generate_candidates(tw, cfg, opt_cls, spacing, walls, keep)
        st.caption(f"후보 위치 {len(cands)}곳 (문·랙·발열설비와 겹치는 곳 제외). 기존 {CLASS_INFO[opt_cls]['label']} 배치는 무시하고 새로 배치합니다.")
    if st.button("▶ 최적화 실행", type="primary"):
        h, ka, kg, lo_, hi_, mg = settings()
        bar = st.progress(0.0, text="준비 중")
        try:
            base = thermal.simulate(tw, ss.devices, ss.bcs, h, ka, kg, (lo_, hi_), label="현재 배치 · 현재 설정온도")
            bar.progress(0.05, text="현재 배치에서 운전온도 최적화")
            span = (0.05, 0.35) if cands is not None else (0.05, 1.0)
            temp = optimize.optimize_temperatures(tw, ss.devices, ss.bcs, cfg, h, ka, kg, lo_, hi_, mg, label="현재 배치 · 최적 운전온도",
                                                  progress=lambda f, m: bar.progress(span[0] + (span[1] - span[0]) * f, text=m))
            place = None
            if cands is not None:
                place = optimize.optimize_placement(tw, ss.devices, ss.bcs, cfg, opt_cls, int(n_dev), cands, h, ka, kg, lo_, hi_, mg,
                                                    objective, progress=lambda f, m: bar.progress(0.35 + 0.65 * f, text=m))
            ss.opt = {"base": base, "temp": temp, "place": place, "cands": cands, "key": model_key(), "band": (lo_, hi_, mg)}
            bar.progress(1.0, text="완료")
        except ValueError as e:
            st.error(str(e))
    opt = ss.opt
    if opt is not None:
        if opt["key"] != model_key():
            st.warning("설정이 바뀌었습니다. 아래 결과는 이전 설정 기준입니다.")
        lo_, hi_, mg = opt["band"]
        rows = []
        for lab, r in (("현재 배치 · 현재 설정온도", opt["base"]), ("현재 배치 · 최적 운전온도", opt["temp"]), ("최적 배치 · 최적 운전온도", opt["place"])):
            if r is None:
                continue
            ok = r.meta.get("zone_in_band", 0) >= 0.9999
            rows.append({"경우": lab, "판정": "✅ 만족" if ok else "❌ 위반", "범위 만족률 [%]": 100 * r.meta.get("zone_in_band", 0),
                         "최저 [°C]": r.meta.get("zone_min"), "최고 [°C]": r.meta.get("zone_max"),
                         "장치 열량 [kW]": device_power_kw(r)})
        best = opt["place"] or opt["temp"]
        if best.meta.get("zone_in_band", 0) >= 0.9999:
            msg = f"✅ **{best.meta['label']}**: 관리구역 전체가 {lo_:.1f}–{hi_:.1f} °C를 만족합니다. 필요한 장치 열량 **{device_power_kw(best):.2f} kW**"
            if opt["place"] is not None and opt["temp"].meta.get("zone_in_band", 0) < 0.9999:
                msg += f" — 현재 배치로는 운전온도를 아무리 조정해도 {100 * opt['temp'].meta['zone_in_band']:.1f}%만 만족합니다. **배치가 문제**였습니다."
            elif opt["place"] is not None:
                saved = 1 - device_power_kw(best) / max(device_power_kw(opt["temp"]), 1e-9)
                msg += f" (현재 배치 최적 운전 대비 에너지 {100 * saved:+.0f}% 절감)" if saved > 0 else ""
            st.success(msg)
        else:
            st.warning(f"⚠️ 범위를 완전히 만족하지 못합니다. 최대 위반량 {best.meta.get('lp_violation', 0):.2f} °C. "
                       "장치 수를 늘리거나, 장치 허용 온도 범위·후보 위치를 넓혀 보세요.")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                     column_config={"범위 만족률 [%]": st.column_config.NumberColumn(format="%.1f"),
                                    "최저 [°C]": st.column_config.NumberColumn(format="%.1f"),
                                    "최고 [°C]": st.column_config.NumberColumn(format="%.1f"),
                                    "장치 열량 [kW]": st.column_config.NumberColumn(format="%.2f")})
        show = st.radio("자세히 볼 결과", [r["경우"] for r in rows], index=len(rows) - 1, horizontal=True)
        res = {"현재 배치 · 현재 설정온도": opt["base"], "현재 배치 · 최적 운전온도": opt["temp"], "최적 배치 · 최적 운전온도": opt["place"]}[show]
        a, b = st.columns([1.25, 1])
        with a:
            g = res.grid
            zc = g.origin[2] + (np.arange(g.shape[2]) + 0.5) * g.h
            zi = int(np.argmin(np.abs(zc - 1.6)))
            chosen = res.meta.get("placement", {}).get("chosen", []) if show.startswith("최적 배치") else []
            plot(viz.fig_topview(tw, [d for d in res.devices if d.get("enabled", True)], opt["cands"] if show.startswith("최적 배치") else None,
                                 chosen, res.T, g.origin, g.h, zi, lo_, hi_, height=560,
                                 title=f"{show} — 높이 {zc[zi]:.2f} m 온도"), key="opt_top")
        with b:
            bh = res.meta["body_heat_W"]
            wall_ko = {"wall_south": "남측 벽", "wall_north": "북측 벽", "wall_west": "서측 벽", "wall_east": "동측 벽"}

            def note(d):
                q = bh.get(d["id"], 0)
                spec = th_cfg["device_types"][d["class"]]
                if abs(q) < 80:
                    return "사실상 대기 → 줄여도 됨"
                if d["T_set"] >= spec["T_max"] - 0.05:
                    return "허용 상한 도달"
                if d["T_set"] <= spec["T_min"] + 0.05:
                    return "허용 하한 도달"
                return ""
            dev_rows = [{"ID": d["id"], "위치": f"{wall_ko.get(d.get('wall'), '')} ({d['x']:.1f}, {d['y']:.1f})",
                         "운전온도 [°C]": d["T_set"], "열량 [kW]": bh.get(d["id"], 0) / 1000, "비고": note(d)}
                        for d in res.devices if d.get("enabled", True)]
            st.markdown("**장치별 운전 조건**")
            st.dataframe(pd.DataFrame(dev_rows), hide_index=True, width="stretch",
                         column_config={"운전온도 [°C]": st.column_config.NumberColumn(format="%.1f"),
                                        "열량 [kW]": st.column_config.NumberColumn(format="%.2f")})
            plot(viz.fig_zone_hist(res.T, res.grid.zone, lo_, hi_, height=230), key="opt_hist")
            c1, c2 = st.columns(2)
            if c1.button("이 배치·온도를 ④ 장치 표에 적용", width="stretch"):
                set_devices(copy.deepcopy(res.devices))
                st.success("적용했습니다. ④/⑤에서 확인하세요.")
            if c2.button("💾 이 결과 저장", width="stretch"):
                p = io.save_result(res, RESULTS, f"{ss.preset_sel}_{show.replace(' · ', '_')}")
                st.success(f"저장됨: {p.name}")
        if res.meta.get("placement"):
            with st.expander("배치 탐색 과정"):
                pl = res.meta["placement"]
                st.caption(f"후보 {pl['n_candidates']}곳 중 {pl['n_devices']}개 선택 · 기준: {pl['objective']} · 탐색 {pl['search_seconds']} s. "
                           "위반량이 음수이면 그만큼 여유를 두고 범위를 만족한다는 뜻입니다.")
                st.dataframe(pd.DataFrame([{"단계": x["step"], "선택된 후보": ", ".join(map(str, x["set"])),
                                            "위반량 [°C]": x["violation"], "추정 열량 [kW]": x["power_kW"]} for x in pl["history"]]),
                             hide_index=True, width="stretch")
        st.subheader("단면 보기")
        slice_viewer(res.T, res.grid.origin, res.grid.h, res.grid.zone, [d for d in res.devices if d.get("enabled", True)], "opt", lo_, hi_)


# =============================================================================== ⑦ result viewer

elif page == PAGES[7]:
    st.header("⑦ 결과 뷰어")
    st.caption("저장된 시뮬레이션 결과(results/*.npz)를 불러와 단면을 자르며 확인합니다. 파일에는 3D 온도장, voxel 정보, 장치·경계조건 설정이 함께 들어 있습니다.")
    files = io.list_results(RESULTS)
    if not files:
        st.info("저장된 결과가 없습니다. ⑤ 또는 ⑥에서 '결과 저장'을 누르세요.")
    else:
        sel = st.selectbox("결과 파일", files, format_func=lambda p: p.name)
        res = get_result(str(sel), sel.stat().st_mtime)
        tg = res.meta.get("target", {"low": low, "high": high})
        lo_, hi_ = tg["low"], tg["high"]
        st.caption(f"저장 시각 {res.meta.get('saved_at', '?')} · 목표 {lo_}–{hi_} °C · 격자 {res.meta.get('grid', {}).get('shape')}")
        result_metrics(res, lo_, hi_)
        on = [d for d in res.devices if d.get("enabled", True)]
        slice_viewer(res.T, res.grid.origin, res.grid.h, res.grid.zone, on, "viewer", lo_, hi_)
        with st.expander("저장된 설정 (장치 · 경계조건)"):
            c1, c2 = st.columns(2)
            c1.dataframe(pd.DataFrame([{k: d.get(k) for k in ("id", "class", "enabled", "x", "y", "z", "T_set")} for d in res.devices]), hide_index=True)
            c2.dataframe(pd.DataFrame([{k: b.get(k) for k in ("label", "type", "T", "U")} for b in res.bcs]), hide_index=True)
        if len(files) > 1 and st.toggle("다른 결과와 나란히 비교"):
            other = st.selectbox("비교 대상", [f for f in files if f != sel], format_func=lambda p: p.name)
            res2 = get_result(str(other), other.stat().st_mtime)
            if res2.T.shape != res.T.shape:
                st.warning("격자 크기가 달라 비교할 수 없습니다.")
            else:
                g = res.grid
                zc = g.origin[2] + (np.arange(g.shape[2]) + 0.5) * g.h
                z = st.slider("비교 높이 [m]", float(zc[0]), float(zc[-1]), float(zc[np.argmin(np.abs(zc - 1.6))]), float(g.h))
                zi = int(np.argmin(np.abs(zc - z)))
                tw = twin_eff()
                c1, c2 = st.columns(2)
                with c1:
                    plot(viz.fig_topview(tw, on, T=res.T, origin=g.origin, h=g.h, z_index=zi, low=lo_, high=hi_, height=480, title=sel.stem), key="cmp1")
                with c2:
                    plot(viz.fig_topview(tw, [d for d in res2.devices if d.get("enabled", True)], T=res2.T, origin=g.origin, h=g.h, z_index=zi,
                                         low=lo_, high=hi_, height=480, title=other.stem), key="cmp2")
