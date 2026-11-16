"""Real GPU-worker check using saved images. Never imports robot adapters."""
from dataclasses import replace
import json
import multiprocessing as mp
from pathlib import Path
import time

import cv2
import numpy as np

from inference_settings import InferenceSettings
from pi05_policy import policy_worker

ROOT = Path(__file__).parent


def main():
    source = ROOT.parent / "pi05模型诊断_20261020/approach_2step_145002"
    state = json.loads((source / "result.json").read_text())["initial_state"]
    images = {name: cv2.cvtColor(cv2.imread(str(source / f"before_{name}.jpg")), cv2.COLOR_BGR2RGB)
              for name in ("global", "wrist")}
    ctx = mp.get_context("spawn")
    requests, replies = ctx.Queue(), ctx.Queue()
    worker = ctx.Process(target=policy_worker, args=(requests, replies))
    worker.start()

    def wait_for(kind):
        while True:
            reply = replies.get(timeout=120)
            if reply["kind"] == "progress":
                print(reply["text"], flush=True)
            elif reply["kind"] == "error":
                raise RuntimeError(reply["trace"])
            elif reply["kind"] == kind:
                return reply

    try:
        model = next((ROOT.parent / "模型下载_20261020").glob("pi05*/*.pt"))
        requests.put(dict(kind="load", token=1, path=str(model)))
        loaded = wait_for("loaded")
        assert len(loaded["verification"]["sha256"]) == 64
        options = InferenceSettings()
        outputs, results = [], []
        for index, selected in enumerate((options, options, replace(options, denoise_steps=5),
                                          replace(options, seed=11))):
            requests.put(dict(kind="infer", token=index + 2, **images, state=state,
                              prompt=(ROOT / "assets/task.txt").read_text(),
                              settings=selected.to_dict()))
            reply = wait_for("actions")
            actions = np.asarray(reply["actions"])
            assert actions.shape == (50, 8) and np.isfinite(actions).all()
            outputs.append(actions)
            results.append(dict(settings=selected.to_dict(), elapsed_ms=reply["elapsed"]*1000,
                                token=reply["token"], shape=list(actions.shape),
                                first_action=actions[0].tolist()))
        repeat_error = float(np.abs(outputs[0] - outputs[1]).max())
        assert repeat_error < 1e-5
        assert np.max(np.abs(outputs[0] - outputs[2])) > 1e-5
        assert np.max(np.abs(outputs[0] - outputs[3])) > 1e-5
        report = dict(time=time.time(), hardware_connected=False, model=str(model),
                      model_verification=loaded["verification"],
                      image_source=str(source), samples=results, repeat_max_abs=repeat_error,
                      note="保存的 JPEG 图像用于参数传递与可重复性检查，不是原始相机帧或任务成功率测试。")
        (ROOT / "configured_inference_check.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(json.dumps(dict(success=True, repeat_max_abs=repeat_error,
                              elapsed_ms=[r["elapsed_ms"] for r in results])), flush=True)
    finally:
        requests.put(dict(kind="quit"))
        worker.join(timeout=5)
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=2)


if __name__ == "__main__":
    main()
