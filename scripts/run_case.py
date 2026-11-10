"""Run one thermal case from the command line and save the result (for batch experiments / thesis figures).

examples
    python scripts/run_case.py --preset winter --mode sim
    python scripts/run_case.py --preset winter --mode temp            # current layout, optimal temperatures
    python scripts/run_case.py --preset winter --mode place --n 4     # optimal layout + temperatures
    python scripts/run_case.py --preset summer --mode place --cls air_conditioner --n 4 --objective energy

Device / BC edits: pass a JSON file exported from your own code with --devices / --bcs
(lists of dicts with the same keys as data/twin.json devices and the BC table).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from thermal_twin import io, optimize, thermal, twin  # noqa: E402
from thermal_twin.scene import load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "warehouse.yaml"))
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--preset", default="winter")
    ap.add_argument("--mode", choices=["sim", "temp", "place"], default="sim")
    ap.add_argument("--cls", default=None, help="device class to place (default: first active class of the preset)")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--spacing", type=float, default=2.0)
    ap.add_argument("--objective", choices=["margin", "energy"], default="margin")
    ap.add_argument("--low", type=float, default=None)
    ap.add_argument("--high", type=float, default=None)
    ap.add_argument("--margin", type=float, default=None)
    ap.add_argument("--h", type=float, default=None)
    ap.add_argument("--devices", default=None, help="JSON file with a device list")
    ap.add_argument("--bcs", default=None, help="JSON file with a boundary-condition list")
    ap.add_argument("--label", default=None)
    a = ap.parse_args()

    cfg = load_config(a.config)
    th = cfg["thermal"]
    tw = io.load_twin(Path(a.data))
    devs = json.load(open(a.devices, encoding="utf-8")) if a.devices else twin.preset_devices(tw, cfg, a.preset)
    bcs = json.load(open(a.bcs, encoding="utf-8")) if a.bcs else twin.default_boundaries(tw, cfg, a.preset)
    low = a.low if a.low is not None else th["target"]["low"]
    high = a.high if a.high is not None else th["target"]["high"]
    margin = a.margin if a.margin is not None else th["target"]["margin"]
    h = a.h or th["grid_h"]
    ka, kg = th["materials"]["air"], th["materials"]["goods"]
    log = lambda f, m: print(f"  [{100 * f:5.1f}%] {m}")

    if a.mode == "sim":
        res = thermal.simulate(tw, devs, bcs, h, ka, kg, (low, high), label="simulation")
    elif a.mode == "temp":
        res = optimize.optimize_temperatures(tw, devs, bcs, cfg, h, ka, kg, low, high, margin, progress=log)
    else:
        cls = a.cls or th["presets"][a.preset]["active_device_classes"][0]
        keep = [d for d in devs if d["class"] != cls and d.get("enabled", True)]
        cands = optimize.generate_candidates(tw, cfg, cls, a.spacing, None, keep)
        res = optimize.optimize_placement(tw, devs, bcs, cfg, cls, a.n, cands, h, ka, kg, low, high, margin, a.objective, progress=log)

    m = res.meta
    print(f"\n관리구역 {low}-{high} °C 만족률 {100 * m['zone_in_band']:.1f}%  (최저 {m['zone_min']} / 평균 {m['zone_mean']} / 최고 {m['zone_max']} °C)")
    for d in res.devices:
        if d.get("enabled", True):
            print(f"  {d['id']:7s} ({d['x']:5.1f}, {d['y']:5.1f})  {d['T_set']:6.1f} °C  {m['body_heat_W'].get(d['id'], 0) / 1000:6.2f} kW")
    path = io.save_result(res, Path(a.out), a.label or f"{a.preset}_{a.mode}")
    print(f"\n저장: {path}")


if __name__ == "__main__":
    main()
