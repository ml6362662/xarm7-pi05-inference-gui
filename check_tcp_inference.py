"""Real checkpoint + recorded images smoke check; never imports the robot SDK."""
import argparse
import gc
import gzip
import json
from pathlib import Path
import time

from pi05_policy import Pi05Policy, ROOT
from model_contract import TCP, tcp_action_to_sdk, unit_wxyz
from inference_settings import InferenceSettings
import cv2
import numpy as np
import pyarrow.parquet as pq
import torch


def samples():
    download = ROOT.parent / "模型下载_TCP_20261021"
    payload = json.loads(gzip.decompress((download / "training_snapshot/artifacts/tcp_payload.json.gz").read_bytes()))
    ep = payload[0]["episodes"][0]
    data = ROOT.parent / "数据导出_20261019/xarm7_bread_cup_success_lerobot21_14fps"
    manifest = json.loads((data / "export_manifest.json").read_text())["episodes"][0]
    assert manifest["source"] == ep["source"]
    table = pq.read_table(data / "data/chunk-000/episode_000000.parquet")
    joints = np.asarray([r["joint_state"] for r in ep["frames"]], dtype=np.float32)
    np.testing.assert_allclose(joints, table["observation.state"].to_pylist(), atol=1e-6)
    states = np.asarray([r["state"] for r in ep["frames"]], dtype=np.float32)
    # Reproduce training-window sign continuity, including initial w >= 0.
    for window in manifest["training_windows"]:
        reference = None
        for index in range(window["from_frame"], window["to_frame"]):
            states[index, 3:7] = unit_wxyz(states[index, 3:7], reference)
            reference = states[index, 3:7]
    out = []
    for frame in (0, 50, 100):
        images = []
        for camera in ("cam_high", "cam_left_wrist"):
            cap = cv2.VideoCapture(str(data / f"videos/chunk-000/observation.images.{camera}/episode_000000.mp4"))
            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
                ok, bgr = cap.read()
                if not ok: raise RuntimeError("无法读取录制图像")
                images.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            finally:
                cap.release()
        out.append((frame, images, states[frame], joints[frame]))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, action="append")
    parser.add_argument("--output", type=Path, default=ROOT / "tcp_inference_check.json")
    args = parser.parse_args()
    models = args.model or sorted((ROOT.parent / "模型下载_TCP_20261021").glob("pi05*/*.pt"))
    recorded = samples()
    report = dict(hardware_connected=False, purpose="recorded-frame smoke check, not task success evaluation", models=[])
    options = InferenceSettings(denoise_steps=2).to_dict()
    for filename in models:
        start = time.monotonic()
        policy = Pi05Policy(filename, lambda s: print(s, flush=True))
        item = dict(model=str(filename), mode=policy.contract["mode"], load_seconds=time.monotonic()-start, samples=[])
        for frame, images, tcp, joints in recorded:
            state = tcp if policy.contract["mode"] == TCP else joints
            actions, elapsed = policy.predict(*images, state, (ROOT / "assets/task.txt").read_text(), options)
            assert actions.shape == (50, 8) and np.isfinite(actions).all()
            entry = dict(frame=frame, shape=list(actions.shape), finite=True,
                         seconds=elapsed, first_action=actions[0].tolist())
            if policy.contract["mode"] == TCP:
                np.testing.assert_allclose(np.linalg.norm(actions[:, 3:7], axis=1), 1, atol=1e-6)
                for action in actions:
                    assert np.isfinite(tcp_action_to_sdk(action)).all()
                entry.update(quaternion_unit=True, first_sdk_pose=tcp_action_to_sdk(actions[0]).tolist())
            item["samples"].append(entry)
        report["models"].append(item)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(json.dumps(item, ensure_ascii=False), flush=True)
        del policy
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
