"""Outcome workflow checks; never connect devices or load a GPU model."""
import json
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import pytest

import engine
from run_recording import RunRecorder
import run_results
from run_results import annotate, list_recordings, read_result, write_json
from test_control import bare_engine
from test_ui import window


def startable_engine():
    e = bare_engine()
    e.running = False
    e.pending = None
    e.model_path = "example.pt"
    e.model_verification = {"test": True}
    e.observations = lambda: (None, None, np.zeros(8))
    calls = []
    e.robot = SimpleNamespace(
        begin_segment=lambda options: {"state": [0.] * 8},
        start=lambda allowed: calls.append(("start", allowed())),
        stop=lambda: calls.append(("stop", True)),
    )
    return e, calls


def test_recording_opt_out_does_not_construct_recorder(monkeypatch):
    e, calls = startable_engine()
    monkeypatch.setattr(engine, "RunRecorder", lambda *a, **k: pytest.fail("opt-out wrote detailed recording"))
    e.command(dict(kind="start", token=2, prompt="test", record_run=False))
    assert e.running and e.recorder is None and calls == [("start", True)]
    e.record("action", {"raw": [0] * 8})
    e.request("pause")
    e.command(e.commands.get_nowait())
    assert not e.running
    assert not any("completed_recording" in ev for ev in list(e.events.queue))


def test_recording_choice_validation_precedes_hardware():
    e, _ = startable_engine()
    e.observations = lambda: pytest.fail("invalid flag reached hardware")
    with pytest.raises(ValueError):
        e.command(dict(kind="start", token=2, prompt="test", record_run="false"))


def test_recorded_run_stops_before_finalizing_and_emits_annotation_path(monkeypatch, tmp_path):
    e, calls = startable_engine()
    monkeypatch.setattr(engine, "RunRecorder", lambda data: RunRecorder(data, tmp_path))
    e.command(dict(kind="start", token=2, prompt="test", record_run=True))
    recorder = e.recorder
    assert read_result(recorder.path)["outcome"] == "pending"
    e.record("prediction", {"actions": [[0.] * 8] * 50})
    finish = recorder.finish
    def checked_finish(reason):
        assert calls[-1][0] == "stop"
        assert not e.running and not e.actions and e.cancel.is_set()
        return finish(reason)
    recorder.finish = checked_finish
    e.request("pause")
    e.command(e.commands.get_nowait())
    result = read_result(recorder.path)
    assert result["stop_reason"] == "用户暂停"
    assert result["run_state"] == "ended" and result["ended_at"]
    assert result["outcome"] == "pending" and not result["annotation_history"]
    events = list(e.events.queue)
    assert sum(ev.get("completed_recording") == str(recorder.path) for ev in events) == 1
    e.halt()
    assert sum("completed_recording" in ev for ev in list(e.events.queue)) == 1


@pytest.mark.parametrize("outcome", ["success", "failure"])
def test_label_persists_without_altering_trace_and_can_be_edited(tmp_path, outcome):
    r = RunRecorder({"prompt": "test"}, tmp_path)
    r.record("observation", {"state": [0.] * 8}, {"global_rgb": np.zeros((4, 4, 3), np.uint8)})
    r.record("prediction", {"actions": [[0.] * 8] * 50})
    assert r.finish("用户暂停") == ""
    trace = (r.path / "trace.jsonl").read_bytes()
    result = annotate(r.path, outcome, "  碰到纸杯  ", tmp_path)
    assert result["outcome"] == outcome and result["note"] == "碰到纸杯"
    annotate(r.path, "failure", "修正说明", tmp_path)
    r.finish("重复结束")
    saved = read_result(r.path)
    assert saved["note"] == "修正说明" and len(saved["annotation_history"]) == 2
    assert (r.path / "trace.jsonl").read_bytes() == trace
    assert list(r.path.glob("*.jpg"))
    assert list_recordings(tmp_path)[0]["result"] == saved


def test_active_recording_cannot_be_labeled(tmp_path):
    r = RunRecorder({}, tmp_path)
    try:
        with pytest.raises(ValueError, match="尚未结束"):
            annotate(r.path, "success", root=tmp_path)
    finally:
        r.finish()


def test_interrupted_record_can_be_labeled_without_fabricated_end_time(tmp_path, monkeypatch):
    p = tmp_path / "run_interrupted"
    p.mkdir()
    write_json(p / "result.json", dict(outcome="pending", run_state="recording",
                                     owner_pid=999999, ended_at=None))
    def gone(*args):
        raise ProcessLookupError()
    monkeypatch.setattr(run_results.os, "kill", gone)
    result = annotate(p, "failure", "程序退出", tmp_path)
    assert result["run_state"] == "interrupted" and result["ended_at"] is None


def test_annotation_write_failure_keeps_previous_result(tmp_path, monkeypatch):
    r = RunRecorder({}, tmp_path)
    r.finish()
    annotate(r.path, "failure", "旧备注", tmp_path)
    old = (r.path / "result.json").read_bytes()
    def fail(*args):
        raise OSError("磁盘不可写")
    monkeypatch.setattr(run_results.os, "replace", fail)
    with pytest.raises(OSError):
        annotate(r.path, "success", "新备注", tmp_path)
    assert (r.path / "result.json").read_bytes() == old
    assert not list(r.path.glob(".*.tmp"))


def test_bad_label_and_outside_path_rejected(tmp_path):
    r = RunRecorder({}, tmp_path)
    r.finish()
    for label in ("pending", "unknown"):
        with pytest.raises(ValueError):
            annotate(r.path, label, root=tmp_path)
    with pytest.raises(ValueError):
        annotate(r.path, "success", root=tmp_path / "different")
    with pytest.raises(ValueError):
        annotate(r.path, "failure", "字" * 2001, root=tmp_path)


def test_broken_record_does_not_hide_other_runs(tmp_path):
    broken = tmp_path / "run_broken"
    broken.mkdir()
    (broken / "result.json").write_text("{bad")
    r = RunRecorder({}, tmp_path)
    r.finish()
    entries = list_recordings(tmp_path)
    assert len(entries) == 2
    assert sum(bool(x["error"]) for x in entries) == 1


def test_failure_finishes_record_without_auto_label_and_even_if_stop_raises(tmp_path):
    e, _ = startable_engine()
    e.recorder = RunRecorder({}, tmp_path)
    path = e.recorder.path
    def stop_failed():
        raise RuntimeError("停止失败")
    e.robot.stop = stop_failed
    with pytest.raises(RuntimeError, match="停止失败"):
        e.halt("相机反馈失效")
    result = read_result(path)
    assert result["stop_reason"] == "相机反馈失效"
    assert result["outcome"] == "pending"


def test_ui_passes_record_choice_and_locks_during_run(window):
    window.record_run.setChecked(False)
    window.start_inference()
    assert window._test_requests[-1][1]["record_run"] is False
    assert not window.record_run.isEnabled()
    window.starting = False
    window.running = True
    window.buttons()
    assert not window.review_records.isEnabled()
    assert window.pause.isEnabled()


def test_label_dialog_saves_selected_run_not_latest(window, tmp_path):
    from recording_dialog import RecordingDialog
    first, second = RunRecorder({}, tmp_path), RunRecorder({}, tmp_path)
    first.finish()
    second.finish()
    d = RecordingDialog(window, tmp_path)
    d.refresh(first.path)
    d.outcome.setCurrentIndex(d.outcome.findData("failure"))
    d.note.setPlainText("未夹住面包")
    d.save()
    assert read_result(first.path)["outcome"] == "failure"
    assert read_result(second.path)["outcome"] == "pending"
    d.refresh(first.path)
    assert d.note.toPlainText() == "未夹住面包"
    assert not window._test_requests
    d.close()


def test_ui_shows_annotation_after_stop_without_blocking_heartbeat(window, monkeypatch, tmp_path):
    import app as ui
    from recording_dialog import RecordingDialog
    r = RunRecorder({}, tmp_path)
    r.finish()
    monkeypatch.setattr(ui, "RecordingDialog", lambda parent: RecordingDialog(parent, tmp_path))
    window.engine.events.put(dict(text="记录完成", loaded=False, loading=False,
                                 connected=False, running=False, completed_recording=str(r.path)))
    window.poll()
    assert window.record_dialog.isVisible()
    assert not window.record_dialog.isModal()
    assert window.record_dialog.selected_path == r.path
    assert not window._test_requests
    window.engine.heartbeat = 0
    window.poll()
    assert window.engine.heartbeat > 0
    window.record_dialog.close()
    window.poll()
    assert not window.record_dialog.isVisible()
    assert read_result(r.path)["outcome"] == "pending"


def test_ui_does_not_open_annotation_during_gripper_or_shutdown(window):
    for field in ("busy_gripper", "busy_home", "closing"):
        setattr(window, field, True)
        window.completed_recordings = ["/unopened"]
        window.poll()
        assert window.record_dialog is None
        setattr(window, field, False)
    window.completed_recordings.clear()
