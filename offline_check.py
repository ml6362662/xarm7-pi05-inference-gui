#!/usr/bin/env python3
"""Real CUDA inference on exported frames. Does not import or connect robot SDK."""
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
import pyarrow.parquet as pq

from pi05_policy import Pi05Policy, ROOT
from model_contract import (TCP, EEFDELTA, POSE_MODES, tcp_action_to_sdk,
                            tcp_state_to_eefdelta, decode_eefdelta_actions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path)
    args = parser.parse_args()
    model_path = args.model or next((ROOT.parent / "模型下载_20261020").glob("pi05*/*.pt"))
    data = ROOT.parent / "数据导出_20261019/xarm7_bread_cup_success_lerobot21_14fps"
    table = pq.read_table(data / "data/chunk-000/episode_000000.parquet")
    samples = []
    for frame in (0, 50, 100):
        images = []
        for camera in ("cam_high", "cam_left_wrist"):
            cap = cv2.VideoCapture(str(data / f"videos/chunk-000/observation.images.{camera}/episode_000000.mp4"))
            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
                ok, image = cap.read()
                if not ok:
                    raise RuntimeError("读取离线测试视频失败")
                images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            finally:
                cap.release()
        samples.append((frame, images, table["observation.state"][frame].as_py()))
    started = time.monotonic()
    model = Pi05Policy(model_path, lambda text: print(text, flush=True))
    loaded_in = time.monotonic() - started
    if model.contract["mode"] in POSE_MODES:
        from check_tcp_inference import samples as tcp_samples
        samples = [(frame, images, tcp_state_to_eefdelta(state) if model.contract["mode"] == EEFDELTA else state)
                   for frame, images, state, _ in tcp_samples()]
    results = []
    for frame, images, state in samples:
        actions, elapsed = model.predict(*images, state, (ROOT / "assets/task.txt").read_text())
        r = dict(frame=frame, shape=list(actions.shape), finite=bool(np.isfinite(actions).all()),
                 inference_seconds=elapsed, first_action=actions[0].tolist(),
                 gripper_min=float(actions[:, -1].min()), gripper_max=float(actions[:, -1].max()))
        if model.contract["mode"] in POSE_MODES:
            decoded = decode_eefdelta_actions(state, actions) if model.contract["mode"] == EEFDELTA else actions
            r["first_sdk_pose_mm_rad"] = tcp_action_to_sdk(decoded[0]).tolist()
        else:
            r["first_joint_delta_rad"] = (actions[0, :7] - np.asarray(state)[:7]).tolist()
        results.append(r)
        print(json.dumps(r), flush=True)
    report = dict(model=str(model_path), model_contract=model.contract, strict_state_dict=True, load_and_warmup_seconds=loaded_in,
                  samples=results, hardware_connected=False,
                  note="Training-data smoke test; not held-out accuracy or real-robot success rate.")
    (ROOT / "offline_check_result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
