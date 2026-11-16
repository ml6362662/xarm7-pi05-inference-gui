"""No hardware connections: exercise the configurable production path."""
from dataclasses import replace
import json
import queue
import sys
import time
from types import SimpleNamespace
from contextlib import nullcontext

import numpy as np
import pytest

from hardware import Cancelled, Robot, SegmentComplete, bounded_target
from inference_settings import InferenceSettings, load_settings, preset, save_settings, simple_settings
from run_recording import RunRecorder
from test_control import bare_engine, policy_robot


def configured_robot():
    r, sent, snapshot, writes = policy_robot()
    r.motion_owned = True
    r.origin_q = np.zeros(7)
    r.origin_pose = np.array([0., 0., 500., 0., 0., 0.])
    r.path_mm = 0.
    r.floor = None
    r.fk = lambda q: np.r_[np.asarray(q[:3]) * 1000 + [0, 0, 500], q[3:6]]
    return r, sent, snapshot, writes


def test_presets_match_recorded_probe_parameters():
    c = preset("approach")
    assert (c.denoise_steps, c.execute_frames, c.seed, c.fixed_seed) == (2, 4, 7, True)
    assert (c.control_hz, c.joint_step_speed, c.servo_acceleration, c.tcp_step_mm) == (14, .1, .3, 1)
    assert (c.segment_displacement_mm, c.segment_path_mm, c.segment_joint_deg,
            c.segment_orientation_deg, c.segment_seconds, c.segment_cycles) == (50, 60, 7, 10, 12, 30)
    small = preset("small")
    assert (small.segment_displacement_mm, small.segment_path_mm, small.segment_joint_deg,
            small.segment_orientation_deg, small.segment_seconds, small.segment_cycles) == (20, 22, 3, 5, 4, 6)
    assert preset("lift").gripper_mode == preset("transfer").gripper_mode == "hold"
    assert preset("auto").gripper_mode == "auto"


@pytest.mark.parametrize("values", [
    {"denoise_steps": 0}, {"execute_frames": 51}, {"denoise_steps": True},
    {"seed": -1}, {"control_hz": float("nan")}, {"fixed_seed": 1},
    {"tcp_step_mm": 0}, {"joint_step_speed": .5}, {"gripper_mode": "unknown"},
    {"close_fraction": .7, "open_fraction": .6},
    {"segment_path_mm": 10}, {"extra_key": 1},
])
def test_invalid_settings_rejected(values):
    with pytest.raises(ValueError):
        InferenceSettings.from_dict(values)


def test_settings_round_trip_and_corruption(tmp_path):
    p = tmp_path / "settings.json"
    settings = replace(preset("auto"), denoise_steps=3, execute_frames=6, fixed_seed=False)
    save_settings(settings, p)
    assert load_settings(p) == settings
    p.write_text("{bad")
    with pytest.raises(ValueError):
        load_settings(p)


def test_review_stops_before_any_closing_or_joint_command():
    r, sent, _, writes = configured_robot()
    with pytest.raises(SegmentComplete, match="确认对齐") as stop:
        r.controlled_action(np.r_[np.zeros(7), .79], lambda: True, preset("approach"))
    assert not sent and not writes
    assert stop.value.trace["raw"][7] == .79


def test_hold_never_opens_a_held_object():
    r, sent, snapshot, writes = configured_robot()
    snapshot["position_mm"] = 5.
    with pytest.raises(SegmentComplete, match="要求打开"):
        r.controlled_action(np.r_[np.zeros(7), .9], lambda: True, preset("lift"))
    assert not sent and not writes
    r.controlled_action(np.r_[np.zeros(7), .05], lambda: True, preset("lift"))
    assert not sent and len(writes) == 1


def test_configured_joint_limiter_matches_staged_probe_math():
    r, sent, _, writes = configured_robot()
    settings = preset("approach")
    prev, velocity = np.zeros(7), np.zeros(7)
    # Same base limiter -> 0.1rad/s cap -> iterative 1mm TCP step cap.
    for sign in [1.] * 5 + [-1.] * 5:
        a = np.r_[np.array([.03, .02 * sign, -.01, 0, 0, 0, 0]), 1.]
        expected, _, _ = bounded_target(a, prev, prev, velocity, r.lower, r.upper)
        delta = np.clip(expected - prev, -.1/14, .1/14)
        for _ in range(12):
            expected = prev + delta
            distance = np.linalg.norm(r.fk(expected)[:3] - r.fk(prev)[:3])
            if distance <= 1:
                break
            delta *= min(.8, 1/distance)
        result = r.controlled_action(a, lambda: True, settings)
        np.testing.assert_allclose(result["sent"], expected)
        np.testing.assert_allclose(r.velocity, delta * 14)
        assert np.linalg.norm(np.array(result["sent_tcp"])[:3] - r.fk(prev)[:3]) <= 1 + 1e-9
        velocity, prev = delta * 14, expected
    assert not sent and len(writes) == 10


def test_raw_close_pose_gap_defers_closure_and_near_pose_requests_transition():
    r, sent, _, writes = configured_robot()
    a = np.r_[[.02, 0, 0, 0, 0, 0, 0], 0.]
    result = r.controlled_action(a, lambda: True, preset("auto"))
    assert result["status"] == "await_close_pose"
    assert result["tcp_gap_mm"] == pytest.approx(20.)
    assert not sent and len(writes) == 1
    r.previous = np.r_[.019, np.zeros(6)]
    r.velocity = np.zeros(7)
    result = r.controlled_action(a, lambda: True, preset("auto"))
    assert result["status"] == "gripper_transition"
    assert result["gripper_target"] == 0.
    assert len(writes) == 1  # Transition itself never advances the arm.


def test_segment_and_height_checks_stop_before_next_command():
    r, sent, _, writes = configured_robot()
    r.path_mm = 60.
    with pytest.raises(SegmentComplete, match="路径"):
        r.controlled_action(np.r_[[.02, 0, 0, 0, 0, 0, 0], 1.], lambda: True, preset("approach"))
    r.path_mm = 0.
    r.floor = 500.
    with pytest.raises(SegmentComplete, match="最低高度"):
        r.controlled_action(np.r_[[0, 0, -.02, 0, 0, 0, 0], 1.], lambda: True, preset("approach"))
    assert not writes and not sent


def test_configured_action_cancellation_prevents_writes():
    r, sent, _, writes = configured_robot()
    checks = iter([True, False])
    with pytest.raises(Cancelled):
        r.controlled_action(np.r_[np.zeros(7), 1.], lambda: next(checks), preset("approach"))
    assert not writes and not sent


def waiting_engine(monkeypatch, resume=True):
    e = bare_engine()
    e.settings = preset("auto")
    clock = [10.]
    monkeypatch.setattr("engine.time.monotonic", lambda: clock[0])
    e.heartbeat = 10.
    e.worker = None
    e.cameras = {}
    e.pending = None
    e.actions.extend([np.zeros(8)] * 3)
    e.segment_deadline = 20.
    e.gripper_wait = dict(target=0., started=10., submitted=10., acknowledged=None, resume=resume, token=2)
    feedback = dict(connected=True, command_fault=False, error="", position_age_s=.1,
                    position_mm=5., committed_target_mm=None, command_time_s=None)
    starts = []
    e.robot = SimpleNamespace(
        gripper=SimpleNamespace(snapshot=lambda: feedback), last_grip=0.,
        health=lambda: None, start=lambda allowed: starts.append(allowed()),
        stop=lambda: None)
    return e, clock, feedback, starts


def test_gripper_wait_requires_ack_then_full_delay_and_discards_cached_actions(monkeypatch):
    e, clock, feedback, starts = waiting_engine(monkeypatch)
    e.tick()
    assert not starts
    feedback.update(committed_target_mm=0., command_time_s=10.2)
    clock[0] = e.heartbeat = 10.5
    e.tick()
    assert not starts
    clock[0] = e.heartbeat = 14.1
    e.tick()
    assert not starts
    clock[0] = e.heartbeat = 14.21
    e.tick()
    assert starts == [True] and e.gripper_wait is None and not e.actions
    assert e.robot.last_grip == 0.
    assert e.segment_deadline == pytest.approx(24.21)


def test_gripper_pending_not_acknowledged_times_out(monkeypatch):
    e, clock, feedback, starts = waiting_engine(monkeypatch)
    # An old matching command is not an acknowledgement of this command.
    feedback.update(committed_target_mm=0., command_time_s=9.)
    clock[0] = e.heartbeat = 11.6
    with pytest.raises(RuntimeError, match="确认下发"):
        e.tick()
    assert not starts


def test_gripper_wait_cancellation_never_restarts_arm(monkeypatch):
    e, _, _, starts = waiting_engine(monkeypatch)
    e.request("pause")
    with pytest.raises(Cancelled):
        e.tick()
    e.halt()
    assert e.gripper_wait is None and not starts


def test_manual_gripper_wait_never_resumes_inference(monkeypatch):
    e, clock, feedback, starts = waiting_engine(monkeypatch, resume=False)
    e.running = False
    feedback.update(committed_target_mm=0., command_time_s=10.)
    clock[0] = e.heartbeat = 14.1
    e.tick()
    assert not starts and e.gripper_wait is None and e.cancel.is_set()


def test_gripper_wait_feedback_failure_blocks_resume(monkeypatch):
    e, _, feedback, starts = waiting_engine(monkeypatch)
    feedback["command_fault"] = True
    with pytest.raises(RuntimeError, match="反馈失效"):
        e.tick()
    assert not starts


def test_gripper_transition_waits_for_stationary_joint_feedback(monkeypatch):
    e, clock, feedback, starts = waiting_engine(monkeypatch)
    e.gripper_wait["submitted"] = None
    velocity = [1.] * 7
    writes = []
    e.robot.arm = SimpleNamespace(get_joint_states=lambda **kw: (0, [np.zeros(7), velocity]))
    e.robot.manual_gripper = lambda aperture, allowed: writes.append((aperture, allowed()))
    e.tick()
    assert not writes
    velocity[:] = [0.] * 7
    clock[0] = e.heartbeat = 10.2
    e.tick()
    assert writes == [(0., True)]
    assert not starts


def test_engine_passes_actual_settings_to_worker_and_limits_horizon():
    e = bare_engine()
    e.settings = replace(preset("approach"), execute_frames=3, denoise_steps=4, seed=11)
    e.worker = None
    e.pending = None
    e.next_action = 0.
    e.cameras = {}
    e.requests = queue.Queue()
    e.prompt = "test"
    state = np.zeros(8)
    e.observations = lambda: (np.zeros((2, 2, 3)), np.zeros((2, 2, 3)), state)
    e.tick()
    request = e.requests.get_nowait()
    assert request["settings"] == e.settings.to_dict()
    e.result(dict(kind="actions", token=2, actions=np.zeros((50, 8)), elapsed=.1))
    assert len(e.actions) == 3 and e.inference_info["execute_frames"] == 3


@pytest.mark.parametrize("frames", [15, 30, 50])
def test_engine_accepts_horizon_beyond_old_limit_without_reordering(frames):
    e = bare_engine()
    e.settings = InferenceSettings.from_dict({"execute_frames": frames})
    actions = np.zeros((50, 8))
    actions[:, 0] = np.arange(50) / 1000
    e.result(dict(kind="actions", token=2, actions=actions, elapsed=.1))
    np.testing.assert_array_equal(np.asarray(e.actions), actions[:frames])
    assert e.inference_info["execute_frames"] == frames


def test_segment_timeout_does_not_request_another_prediction():
    e = bare_engine()
    e.worker = None
    e.segment_deadline = time.monotonic() - 1
    with pytest.raises(SegmentComplete, match="时间"):
        e.tick()


def test_start_rejects_bad_settings_before_touching_robot():
    e = bare_engine()
    e.running = False
    e.pending = None
    e.observations = lambda: pytest.fail("invalid settings reached hardware")
    with pytest.raises(ValueError):
        e.command(dict(kind="start", token=2, prompt="test", settings={"execute_frames": 99}))


def test_configured_inference_pose_drift_limit_is_half_degree():
    e = bare_engine()
    e.settings = preset("approach")
    e.robot.state = lambda: np.r_[.01, np.zeros(7)]  # 0.57°, formerly accepted by 0.06rad guard.
    with pytest.raises(RuntimeError, match="位置已变化"):
        e.result(dict(kind="actions", token=2, actions=np.zeros((50, 8)), elapsed=.1))
    assert not e.actions


def test_hold_segment_origin_floor_is_applied_without_motion():
    r, sent, _, writes = configured_robot()
    r.recovery = SimpleNamespace(home={}, validate_profile=lambda home: None)
    r.arm.get_joint_states = lambda **kw: (0, [np.zeros(7), np.zeros(7)])
    origin = r.begin_segment(preset("lift"))
    assert origin["floor_mm"] == 499.5
    origin = r.begin_segment(replace(preset("lift"), floor_enabled=True, floor_mm=510.))
    assert origin["floor_mm"] == 510.
    assert not writes and not sent


def test_independent_watchdog_stops_stalled_control_thread():
    e = bare_engine()
    calls = []
    waits = iter([False, True])
    e.closing = SimpleNamespace(wait=lambda seconds: next(waits), is_set=lambda: False)
    e.robot.motion_owned = True
    e.robot.stop = lambda: calls.append("stop")
    e.last_control_tick = time.monotonic() - 1.
    e.supervise()
    assert calls == ["stop"] and e.cancel.is_set()


def test_no_gripper_write_if_arm_does_not_stop_in_time(monkeypatch):
    e, clock, _, _ = waiting_engine(monkeypatch)
    e.gripper_wait["submitted"] = None
    e.robot.arm = SimpleNamespace(get_joint_states=lambda **kw: (0, [np.zeros(7), np.ones(7)]))
    e.robot.manual_gripper = lambda *args: pytest.fail("gripper written before stationary")
    clock[0] = e.heartbeat = 12.1
    with pytest.raises(RuntimeError, match="停稳"):
        e.tick()


def test_waiting_close_pose_discards_cached_lift_frames():
    e = bare_engine()
    e.worker = None
    e.pending = None
    e.cameras = {}
    e.next_action = time.monotonic()
    e.actions.extend([np.zeros(8)] * 4)
    e.robot.controlled_action = lambda *args: dict(status="await_close_pose")
    e.tick()
    assert not e.actions


def test_predict_uses_selected_steps_and_reseeds_each_prediction(monkeypatch):
    import pi05_policy as module
    seeds, samples = [], []
    class Tensor:
        def permute(self, *args): return self
        def unsqueeze(self, *args): return self
        def to(self, **kwargs): return self
        def __truediv__(self, other): return self
        def __sub__(self, other): return self
        def __getitem__(self, other): return self
        def float(self): return self
        def cpu(self): return self
        def numpy(self): return np.zeros((50, 32), dtype=np.float32)
    torch = SimpleNamespace(
        from_numpy=lambda *a: Tensor(), as_tensor=lambda *a, **k: Tensor(),
        tensor=lambda *a, **k: Tensor(), float32="float32",
        inference_mode=nullcontext, manual_seed=lambda seed: seeds.append(seed))
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(module, "resize_rgb", lambda x: np.zeros((224, 224, 3), dtype=np.uint8))
    monkeypatch.setattr(module, "normalize_state", lambda *args: np.zeros(8))
    monkeypatch.setattr(module, "prompt_tokens", lambda *args: (np.zeros(200), np.ones(200)))
    p = module.Pi05Policy.__new__(module.Pi05Policy)
    p.contract = {"mode": "joint_absolute"}
    p.stats = {"state": {}, "actions": {"q01": np.zeros(32), "q99": np.ones(32)}}
    p.tokenizer = None
    p.model = SimpleNamespace(sample_actions=lambda **kwargs: samples.append(kwargs) or Tensor())
    options = replace(preset("approach"), denoise_steps=3, seed=11)
    for _ in range(2):
        output, _ = p.predict(None, None, np.zeros(8), "test", options.to_dict())
        assert output.shape == (50, 8)
    assert seeds == [11, 11] and [r["num_steps"] for r in samples] == [3, 3]
    p.predict(None, None, np.zeros(8), "test", replace(options, fixed_seed=False).to_dict())
    assert seeds == [11, 11]
    # Archived scripts explicitly seed torch and override sample_actions;
    # direct calls must not silently replace their chosen seed.
    p.predict(None, None, np.zeros(8), "test")
    assert seeds == [11, 11] and samples[-1]["num_steps"] == 5


def test_run_recording_preserves_settings_predictions_and_images(tmp_path):
    metadata = dict(settings=preset("approach").to_dict(), prompt="test")
    recorder = RunRecorder(metadata, root=tmp_path)
    recorder.record("observation", {"state": [0.] * 8},
                    {"global_rgb": np.zeros((4, 4, 3), dtype=np.uint8)})
    recorder.record("prediction", {"actions": [[0.] * 8] * 50})
    assert recorder.close() == ""
    stored = json.loads((recorder.path / "metadata.json").read_text())
    assert stored["settings"] == metadata["settings"]
    assert {"engine.py", "hardware.py", "pi05_policy.py", "inference_settings.py",
            "run_recording.py", "run_results.py", "recording_dialog.py", "app.py"} <= set(stored["code_hashes"])
    rows = [json.loads(line) for line in (recorder.path / "trace.jsonl").read_text().splitlines()]
    assert len(rows) == 2 and len(rows[1]["actions"]) == 50
    assert (recorder.path / rows[0]["images"]["global_rgb"]).exists()
    assert json.loads((recorder.path / "recording_status.json").read_text())["dropped"] == 0


def test_full_recording_queue_drops_with_counter_instead_of_blocking():
    recorder = RunRecorder.__new__(RunRecorder)
    recorder.error = ""
    recorder.dropped = 0
    recorder.queue = queue.Queue(maxsize=1)
    recorder.record("action", {"sent": [0.] * 7})
    recorder.record("action", {"sent": [0.] * 7})
    assert recorder.dropped == 1 and recorder.queue.qsize() == 1


def test_default_and_migrated_settings_remove_probe_stops():
    for settings in (InferenceSettings(), simple_settings(preset("lift"))):
        assert not settings.segment_enabled
        assert not settings.floor_enabled
        assert not settings.diagnostic_constraints
        assert settings.gripper_mode == "auto"
        assert settings.inference_drift_deg == 3.


def test_continuous_motion_is_not_limited_by_old_tcp_step_or_segment_bounds():
    r, sent, _, writes = configured_robot()
    r.previous = np.array([.08, 0., -.02, .3, 0, 0, 0])
    r.path_mm = 100.
    r.last_grip = 84.
    a = np.r_[[.10, 0., -.02, .3, 0, 0, 0], 1.]
    result = r.controlled_action(a, lambda: True, InferenceSettings())
    assert result["status"] == "motion"
    step_mm = np.linalg.norm(np.array(result["sent_tcp"])[:3] - np.array(result["actual_tcp"])[:3])
    assert step_mm > 3.  # Neither the former 1mm cap nor 3mm test guard applies.
    assert result["path_mm"] > 100.
    assert len(writes) == 1 and not sent


def test_continuous_close_does_not_wait_for_experimental_raw_pose_threshold():
    r, _, _, writes = configured_robot()
    result = r.controlled_action(np.r_[[.02, 0, 0, 0, 0, 0, 0], 0.],
                                 lambda: True, InferenceSettings())
    assert result["tcp_gap_mm"] == pytest.approx(20.)
    assert result["status"] == "gripper_transition" and not writes


def test_restarting_with_already_open_gripper_does_not_reopen_and_wait(monkeypatch):
    r, sent, snapshot, _ = configured_robot()
    snapshot.update(position_mm=83., committed_target_mm=None)
    monkeypatch.setitem(sys.modules, "xarm_quest_relative",
                        SimpleNamespace(prepare_servo_hold=lambda *args: (None, np.zeros(7))))
    r.start(lambda: True)
    result = r.controlled_action(np.r_[np.zeros(7), 1.], lambda: True, InferenceSettings())
    assert r.last_grip == 84. and result["status"] == "motion" and not sent
    snapshot.update(position_mm=5., committed_target_mm=0.)
    r.start(lambda: True)
    result = r.controlled_action(np.r_[np.zeros(7), 0.], lambda: True, InferenceSettings())
    assert r.last_grip == 0. and result["status"] == "motion" and not sent


def test_continuous_mode_keeps_joint_hard_limit_and_invalid_action_checks():
    r, _, _, writes = configured_robot()
    for target in (2., float("nan")):
        with pytest.raises(RuntimeError):
            r.controlled_action(np.r_[[target, 0, 0, 0, 0, 0, 0], 1.],
                                 lambda: True, InferenceSettings())
    assert not writes
