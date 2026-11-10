"""Run the full reconstruction pipeline and cache the results in data/.

    virtual warehouse -> panoramic RGB-D scan -> SAM3 (simulated) + re-projection
    -> TSDF mesh reconstruction -> thermal twin description (twin.json)

usage:  python scripts/build_twin.py [--config configs/warehouse.yaml] [--out data]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from thermal_twin import io, reconstruction, scanner, segmentation, twin  # noqa: E402
from thermal_twin.scene import build_scene, load_config  # noqa: E402


def run(config_path: Path, out: Path, log=print):
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(config_path)
    t0 = time.time()

    def stage(name):
        log(f"[{time.time() - t0:6.1f}s] {name}")

    stage("1/5 가상 현장 생성")
    scene = build_scene(cfg)

    stage("2/5 파노라마 RGB-D 스캔 시뮬레이션")
    scan = scanner.scan(scene)
    io.save_scan(scan, out)
    log(f"         포인트 {len(scan.points):,}개, 스테이션 {len(scan.positions)}곳")

    stage("3/5 SAM3 탐지 + 포인트클라우드 재매핑")
    seg = segmentation.run_segmentation(scan, scene.instances, cfg["sam3"], cfg.get("seed", 0))
    io.save_segmentation(seg, out)
    log(f"         객체 {len(seg.instances)}개, 평균 IoU {seg.stats['mean_iou']}, 누락 {seg.stats['missed']}")

    stage("4/5 TSDF 메쉬 재구성")
    parts, full, mstats = reconstruction.run_reconstruction(scan, seg, cfg["reconstruction"])
    io.save_mesh(parts, full, mstats, out)
    log(f"         삼각형 {mstats['full_faces']:,}개 (표시용 {mstats['display_faces']:,}개)")

    stage("5/5 열 모델(트윈) 생성")
    tw = twin.derive_twin(seg.instances, seg.room, cfg)
    io.save_twin(tw, out)
    log(f"         장치 {len(tw['devices'])}대, 발열설비 {len(tw['heat_sources'])}개, 랙 {len(tw['racks'])}개, 출입문 {len(tw['doors'])}개")
    stage("완료")
    return tw


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "warehouse.yaml"))
    ap.add_argument("--out", default=str(ROOT / "data"))
    a = ap.parse_args()
    run(Path(a.config), Path(a.out))
