"""Bounded asynchronous diagnostic recorder; no SDK or GPU work."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import queue
import threading
import time
from run_results import RECORDINGS_ROOT, timestamp, write_json


class RunRecorder:
    def __init__(self, metadata, root=None):
        root = Path(root) if root is not None else RECORDINGS_ROOT
        self.path = root / ("run_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        self.path.mkdir(parents=True)
        source = Path(__file__).parent
        metadata = dict(metadata, code_hashes={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [source / n for n in ("engine.py", "hardware.py", "pi05_policy.py", "inference_settings.py",
                                           "app.py", "run_recording.py", "run_results.py", "recording_dialog.py", "model_contract.py", "control_checks.py")]
        })
        metadata["recording_enabled"] = True
        write_json(self.path / "metadata.json", metadata)
        self.result = dict(schema_version=1, run_id=self.path.name, started_at=timestamp(),
                           ended_at=None, run_state="recording", owner_pid=os.getpid(),
                           outcome="pending", outcome_label="待标注", note="",
                           annotated_at=None, annotation_history=[], stop_reason="")
        write_json(self.path / "result.json", self.result)
        self.finalized = False
        self.queue = queue.Queue(maxsize=64)
        self.stop = threading.Event()
        self.dropped = 0
        self.error = ""
        self.written = 0
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def record(self, kind, data, images=None):
        if self.error:
            self.dropped += 1
            return
        item = (dict(kind=kind, time=time.time(), monotonic=time.monotonic(), **data), images)
        try:
            self.queue.put_nowait(item)
        except queue.Full:
            self.dropped += 1

    def run(self):
        try:
            with (self.path / "trace.jsonl").open("w") as stream:
                while not self.stop.is_set() or not self.queue.empty():
                    try:
                        row, images = self.queue.get(timeout=.1)
                    except queue.Empty:
                        continue
                    if images:
                        import cv2
                        row["images"] = {}
                        for name, rgb in images.items():
                            filename = f"{self.written:06d}_{name}.jpg"
                            if not cv2.imwrite(str(self.path / filename), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
                                raise RuntimeError("图像记录失败")
                            row["images"][name] = filename
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    stream.flush()
                    self.written += 1
        except Exception as error:
            self.error = str(error)
        finally:
            try:
                (self.path / "recording_status.json").write_text(json.dumps({
                    "written": self.written, "dropped": self.dropped, "error": self.error,
                    "unwritten": self.queue.qsize(),
                }, ensure_ascii=False, indent=2))
            except Exception as error:
                self.error = str(error)

    def close(self):
        self.stop.set()
        self.thread.join(timeout=2)
        if not self.thread.is_alive():
            try:
                (self.path / "recording_status.json").write_text(json.dumps({
                    "written": self.written, "dropped": self.dropped, "error": self.error,
                    "unwritten": self.queue.qsize(),
                }, ensure_ascii=False, indent=2))
            except Exception as error:
                self.error = str(error)
        return self.error or ("记录线程仍在写入" if self.thread.is_alive() else
                              (f"记录队列丢弃 {self.dropped} 条" if self.dropped else ""))

    def finish(self, reason="已停止"):
        """Called only after stopping motion. Never overwrite a later annotation."""
        warning = self.close()
        if not self.finalized:
            self.result.update(ended_at=timestamp(), run_state="ended",
                               stop_reason=reason, recording_warning=warning)
            try:
                write_json(self.path / "result.json", self.result)
                self.finalized = True
            except (OSError, ValueError) as error:
                warning = f"{warning}；结果文件保存失败：{error}".strip("；")
        return warning
