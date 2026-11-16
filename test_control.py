"""Offline checks for the real controller's cancellation and action validation."""
from collections import deque
import json
from pathlib import Path
import queue
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from engine import Engine
from hardware import Cancelled, ResourceLocks, Robot, binary_gripper_target, bounded_target
from pi05_policy import normalize_state, prompt_tokens, read_bundle, resize_rgb
from inference_settings import InferenceSettings


def bare_engine():
    e = Engine.__new__(Engine)
    e.events = queue.Queue()
    e.commands = queue.Queue()
    e.cancel = threading.Event()
    e.closing = threading.Event()
    e.generation = 2
    e.heartbeat = time.monotonic()
    e.loaded, e.loading, e.running, e.connected = True, False, True, True
    e.pending = {"token": 2, "started": time.monotonic(), "state": np.zeros(8)}
    e.actions = deque()
    e.inference_info = None
    e.inference_count = 0
    e.settings = InferenceSettings()
    e.recorder = None
    e.gripper_wait = None
    e.segment_deadline = float("inf")
    e.segment_predictions = 0
    e.last_image_record = -float("inf")
    e.robot = SimpleNamespace(stop=lambda: None, state=lambda: np.zeros(8))
    return e


def test_pause_discards_inflight_result():
    e = bare_engine()
    e.request("pause")
    e.result(dict(kind="actions", token=2, actions=np.ones((50, 8)), elapsed=.2))
    assert not e.actions and e.pending is None
    assert not e.allowed(2)


def test_fresh_result_executes_configured_four_frames():
    e = bare_engine()
    e.result(dict(kind="actions", token=2, actions=np.zeros((50, 8)), elapsed=.2))
    assert len(e.actions) == 4


def test_stale_result_rejected():
    e = bare_engine()
    e.pending["started"] -= 4
    with pytest.raises(RuntimeError, match="过期"):
        e.result(dict(kind="actions", token=2, actions=np.zeros((50, 8)), elapsed=4))
    assert not e.actions


def test_pose_changed_while_inference_rejected():
    e = bare_engine()
    e.robot.state = lambda: np.ones(8)
    with pytest.raises(RuntimeError, match="位置已变化"):
        e.result(dict(kind="actions", token=2, actions=np.zeros((50, 8)), elapsed=.2))


def test_lost_gui_heartbeat_disallows_motion():
    e = bare_engine()
    e.heartbeat -= 2
    assert not e.allowed(2)


def test_pause_invalidates_queued_start():
    e = bare_engine()
    e.running = False
    e.request("pause")
    e.command(dict(kind="start", token=2))
    assert not e.running


def test_resource_lock_conflict_and_release(tmp_path):
    (tmp_path / ".state").mkdir()
    first = ResourceLocks(tmp_path)
    with pytest.raises(RuntimeError, match="占用"):
        ResourceLocks(tmp_path)
    first.close()
    ResourceLocks(tmp_path).close()


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 5.])
def test_invalid_joint_action_rejected(bad):
    a = np.zeros(8)
    a[0] = bad
    with pytest.raises(RuntimeError):
        bounded_target(a, np.zeros(7), np.zeros(7), np.zeros(7), -np.ones(7), np.ones(7))


def test_velocity_and_acceleration_bounded():
    previous = np.zeros(7)
    v = np.zeros(7)
    for _ in range(20):
        target, new_v, grip = bounded_target(
            np.r_[np.full(7, .2), .5], previous, previous, v, -np.ones(7), np.ones(7))
        assert np.max(np.abs(new_v)) <= .35 + 1e-9
        assert np.max(np.abs(new_v-v)) <= .75/14 + 1e-9
        assert np.max(np.abs(target-previous)) <= .35/14 + 1e-9
        assert grip == .5
        previous, v = target, new_v


def test_far_valid_goal_progresses_without_target_gap_pause():
    previous, velocity = np.zeros(7), np.zeros(7)
    for _ in range(40):
        target, new_velocity, _ = bounded_target(
            np.r_[np.full(7, .8), 1.], previous, previous, velocity,
            -np.ones(7), np.ones(7))
        assert np.all(target > previous)
        assert np.max(target - previous) <= .35 / 14 + 1e-9
        assert np.max(np.abs(new_velocity - velocity)) <= .75 / 14 + 1e-9
        previous, velocity = target, new_velocity
    assert np.all(previous > .35)


def test_stalled_arm_still_rejects_advancing_setpoint():
    with pytest.raises(RuntimeError, match="跟随"):
        bounded_target(np.r_[np.full(7, .8), 1.], np.zeros(7),
                       np.full(7, .12), np.zeros(7), -np.ones(7), np.ones(7))


def test_inference_timeout_pauses_without_waiting_for_worker():
    e = bare_engine()
    e.worker = None
    e.cameras = {}
    e.pending["started"] -= 4
    with pytest.raises(RuntimeError, match="计算超时"):
        e.tick()


def test_eight_state_tokens_before_padding():
    captured = []
    tokenizer = SimpleNamespace(encode=lambda text, add_bos: captured.append(text) or [2, 3])
    ids, mask = prompt_tokens(tokenizer, "pick_up\ncup", np.zeros(8))
    assert "State: 128 128 128 128 128 128 128 128;" in captured[0]
    assert "pick up cup" in captured[0]
    assert ids.shape == (200,) and mask.sum() == 2


def test_quantile_state_not_clipped():
    s = {"q01": np.zeros(8), "q99": np.ones(8)}
    assert np.all(normalize_state(np.full(8, 2.), s) > 1)


def test_training_resize_black_letterbox():
    rgb = np.full((480, 640, 3), 255, dtype=np.uint8)
    resized = resize_rgb(rgb)
    assert resized.shape == (224, 224, 3)
    assert np.all(resized[:28] == 0)
    assert np.all(resized[28:196] == 255)
    assert np.all(resized[196:] == 0)


def test_partial_checkpoint_rejected(tmp_path):
    path = tmp_path / "partial.pt"
    path.write_bytes(b"partial")
    cfg = {"actor": {"model": {
        "action_dim": 8, "num_action_chunks": 50, "openpi_data": {"pad_state": False},
        "openpi": {"config_name": "pi05_xarm7_state", "discrete_state_input": True, "num_images_in_input": 2},
    }}}
    (tmp_path / "training_config.yaml").write_text(json.dumps(cfg))
    stats = {name: {"mean": [0]*n, "std": [1]*n, "q01": [-1]*n, "q99": [1]*n}
             for name, n in (("state", 8), ("actions", 32))}
    (tmp_path / "norm_stats.json").write_text(json.dumps({"norm_stats": stats}))
    (tmp_path / "download_verified.json").write_text(json.dumps({"bytes": 99, "sha256": "wrong"}))
    with pytest.raises(ValueError, match="SHA-256"):
        read_bundle(path)


def test_stop_cancels_gripper_and_clears_motion_without_opening():
    writes = []
    r = Robot.__new__(Robot)
    r.gripper = SimpleNamespace(cancel_pending=lambda: writes.append("cancel_gripper"))
    r.arm = SimpleNamespace(set_state=lambda state: writes.append(("state", state)) or 0)
    r.motion_owned = True
    r.stop()
    r.stop()
    assert writes == ["cancel_gripper", ("state", 4), "cancel_gripper"]
    assert not r.motion_owned


def test_home_cancelled_before_any_motion_command():
    r = Robot.__new__(Robot)
    r.gripper = None
    r.arm = SimpleNamespace()
    r.motion_owned = False
    with pytest.raises(Cancelled):
        r.home(lambda: False)


def test_action_cancelled_before_sdk_write():
    r = Robot.__new__(Robot)
    r.arm = SimpleNamespace()
    with pytest.raises(Cancelled):
        r.action(np.zeros(8), lambda: False)


def test_controller_error_does_not_get_cleared():
    r = Robot.__new__(Robot)
    r.arm = SimpleNamespace(get_err_warn_code=lambda: (0, [23, 0]))
    with pytest.raises(RuntimeError, match="故障"):
        r.health()


@pytest.mark.parametrize("state", [0, 1, 2])
def test_servo_accepts_ready_moving_and_idle_feedback(state):
    r = Robot.__new__(Robot)
    r.arm = SimpleNamespace(
        get_err_warn_code=lambda: (0, [0, 0]), get_state=lambda: (0, state), mode=1)
    r.health(moving=True)


@pytest.mark.parametrize("state", [3, 4, 5, 6, 99])
def test_servo_rejects_pause_stop_and_unknown_feedback(state):
    r = Robot.__new__(Robot)
    r.arm = SimpleNamespace(
        get_err_warn_code=lambda: (0, [0, 0]), get_state=lambda: (0, state), mode=1)
    with pytest.raises(RuntimeError, match="停止/保护状态"):
        r.health(moving=True)


def home_fixture(monkeypatch, samples):
    """Run the actual home loop with scripted controller feedback and a fake clock."""
    clock = [0.]
    monkeypatch.setattr("hardware.time.monotonic", lambda: clock[0])
    monkeypatch.setattr("hardware.time.sleep", lambda seconds: clock.__setitem__(0, clock[0]+seconds))
    writes, feedback = [], deque(samples)
    position = [np.zeros(7)]
    last = [samples[-1]]

    def get_state():
        if feedback:
            last[0] = feedback.popleft()
        state, q = last[0]
        position[0] = np.full(7, q)
        return 0, state

    r = Robot.__new__(Robot)
    r.gripper = None
    r.motion_owned = False
    r.health = lambda: None  # State-loop regression; health is tested separately.
    r.arm = SimpleNamespace(
        get_servo_angle=lambda **kw: (0, position[0].tolist()),
        set_state=lambda state: writes.append(("state", state)) or 0,
        set_mode=lambda mode: 0, motion_enable=lambda **kw: 0,
        set_servo_angle=lambda **kw: writes.append(("move", kw)) or 0,
        get_state=get_state,
    )
    r.recovery = SimpleNamespace(
        home={"saved": True}, validate_profile=lambda profile: None,
        plan_home_pose=lambda *args, **kw: {"target": [.1]*7, "max_joint_travel_deg": 6},
    )
    return r, writes, feedback


def test_home_idle_then_moving_then_idle_reaches_target(monkeypatch):
    samples = [(2, 0), (1, .04)] + [(2, .1)] * 5
    r, writes, remaining = home_fixture(monkeypatch, samples)
    r.home(lambda: True)
    assert not remaining
    assert writes[-1] == ("state", 4)
    assert not r.motion_owned


def test_home_already_at_target_accepts_idle(monkeypatch):
    r, writes, _ = home_fixture(monkeypatch, [(2, .1)] * 5)
    r.home(lambda: True)
    assert writes[-1] == ("state", 4)


def test_home_idle_away_from_target_does_not_claim_success(monkeypatch):
    r, _, _ = home_fixture(monkeypatch, [(2, 0)])
    with pytest.raises(RuntimeError, match="返回初始位置超时"):
        r.home(lambda: True)


@pytest.mark.parametrize("state", [3, 4, 5, 6, 99])
def test_home_still_rejects_paused_stopped_unknown_states(monkeypatch, state):
    r, _, _ = home_fixture(monkeypatch, [(state, .1)])
    with pytest.raises(RuntimeError, match=f"state={state}"):
        r.home(lambda: True)


def test_manual_gripper_pauses_policy_and_invalidates_old_prediction():
    e = bare_engine()
    sent = []
    e.robot.manual_gripper = lambda aperture, allowed: sent.append((aperture, allowed))
    e.request("gripper", aperture=0.)
    assert not e.allowed(2)
    e.command(e.commands.get_nowait())
    assert not e.running and not e.actions
    assert sent[0][0] == 0. and sent[0][1]()
    e.result(dict(kind="actions", token=2, actions=np.ones((50, 8)), elapsed=.2))
    assert not e.actions and e.inference_info is None
    e.request("pause")
    assert not sent[0][1]()


def test_start_invalidates_waiting_manual_gripper():
    e = bare_engine()
    sent = []
    e.robot.manual_gripper = lambda aperture, allowed: sent.append(allowed)
    e.request("gripper", aperture=0.)
    e.command(e.commands.get_nowait())
    assert sent[0]()
    e.request("start", prompt="test")
    assert not sent[0]()


def test_superseded_manual_gripper_is_not_sent():
    e = bare_engine()
    e.robot.manual_gripper = lambda *_: pytest.fail("stale manual command was sent")
    e.request("gripper", aperture=84.)
    old = e.commands.get_nowait()
    e.request("pause")
    e.command(old)


def manual_robot():
    r = Robot.__new__(Robot)
    sent = []
    r.settings = {"gripper_speed_mm_s": 50., "gripper_force": 2.}
    snapshot = dict(connected=True, error="", command_fault=False,
                    position_age_s=.1, position_mm=30.)
    r.gripper = SimpleNamespace(
        snapshot=lambda: snapshot,
        submit=lambda *args: sent.append(args),
    )
    # No arm object: manual gripper must not touch arm modes or joint commands.
    return r, sent, snapshot


@pytest.mark.parametrize("aperture", [0., 84.])
def test_manual_gripper_uses_physical_mm_and_collection_settings(aperture):
    r, sent, _ = manual_robot()
    r.manual_gripper(aperture, lambda: True)
    assert sent[0][:3] == (aperture, 50., 2.)


@pytest.mark.parametrize("aperture", [-1., 35., 85., float("nan"), float("inf")])
def test_manual_gripper_invalid_aperture_is_not_sent(aperture):
    r, sent, _ = manual_robot()
    with pytest.raises(ValueError):
        r.manual_gripper(aperture, lambda: True)
    assert not sent


def test_manual_gripper_stale_feedback_is_not_sent():
    r, sent, snapshot = manual_robot()
    snapshot["position_age_s"] = 1.
    with pytest.raises(RuntimeError, match="过期"):
        r.manual_gripper(84., lambda: True)
    assert not sent


def test_manual_gripper_cancelled_is_not_sent():
    r, sent, _ = manual_robot()
    with pytest.raises(Cancelled):
        r.manual_gripper(84., lambda: False)
    assert not sent


def test_prediction_snapshot_keeps_raw_values_and_does_not_queue_logs():
    e = bare_engine()
    a = np.zeros((50, 8))
    a[0, 0], a[0, 7] = np.pi / 6, .5
    e.result(dict(kind="actions", token=2, actions=a, elapsed=.123))
    info = e.inference_info
    assert info["elapsed_ms"] == 123.
    assert info["first_joints_deg"][0] == pytest.approx(30.)
    assert info["first_gripper_mm"] == 42.
    assert info["max_joint_delta_deg"] == pytest.approx(30.)
    assert info["shape"] == (50, 8) and info["execute_frames"] == 4
    assert e.events.empty()
    a[0, 0] = 0  # Display snapshot must not alias a reused action buffer.
    assert info["first_joints_deg"][0] == pytest.approx(30.)


def policy_robot():
    """Actual Robot.action/state with an in-memory SDK and gripper cache."""
    r, sent, snapshot = manual_robot()
    r.health = lambda **kw: None
    r.previous = np.zeros(7)
    r.velocity = np.zeros(7)
    r.lower, r.upper = -np.ones(7), np.ones(7)
    r.last_grip = None
    joints = []
    r.arm = SimpleNamespace(
        get_servo_angle=lambda **kw: (0, r.previous.copy()),
        set_servo_angle_j=lambda q, **kw: joints.append(q) or 0,
    )
    return r, sent, snapshot, joints


def test_binary_policy_hysteresis_only_submits_transitions():
    r, sent, _, joints = policy_robot()
    fractions = [.5, .55, .61, .59, .45, .39, .41, .55, .7, .9]
    for fraction in fractions:
        r.action(np.r_[np.zeros(7), fraction], lambda: True)
    assert [args[0] for args in sent] == [84., 0., 84.]
    assert all(args[1:3] == (50., 2.) for args in sent)
    assert len(joints) == len(fractions)


@pytest.mark.parametrize("fraction,target", [(-.05, 0.), (.4, 0.), (.6, 84.), (1.05, 84.)])
def test_binary_boundaries_and_small_prediction_overshoot(fraction, target):
    assert binary_gripper_target(fraction, None) == target


@pytest.mark.parametrize("fraction", [float("nan"), float("inf"), -.11, 1.11])
def test_binary_rejects_invalid_prediction_before_any_motion(fraction):
    r, sent, _, joints = policy_robot()
    with pytest.raises(RuntimeError):
        r.action(np.r_[np.zeros(7), fraction], lambda: True)
    assert not sent and not joints


def test_initial_deadband_does_not_open_a_held_object():
    r, sent, snapshot, _ = policy_robot()
    snapshot["position_mm"] = 5.
    r.action(np.r_[np.zeros(7), .5], lambda: True)
    assert not sent and r.last_grip is None
    # The model still receives the measured continuous aperture, never binary.
    assert r.state()[7] == pytest.approx(5/84)


def test_gripper_submit_failure_does_not_advance_binary_latch():
    r, sent, _, _ = policy_robot()
    def fail(*args):
        raise RuntimeError("send failed")
    r.gripper.submit = fail
    with pytest.raises(RuntimeError, match="send failed"):
        r.action(np.r_[np.zeros(7), 0.], lambda: True)
    assert r.last_grip is None
    assert not sent


def test_cancel_after_arm_command_prevents_gripper_transition():
    r, sent, _, _ = policy_robot()
    checks = iter([True, True, False])
    with pytest.raises(Cancelled):
        r.action(np.r_[np.zeros(7), 0.], lambda: next(checks))
    assert not sent and r.last_grip is None


def test_gripper_command_fault_stops_policy_before_next_joint_command():
    r, sent, snapshot, joints = policy_robot()
    snapshot["command_fault"] = True
    with pytest.raises(RuntimeError):
        r.action(np.zeros(8), lambda: True)
    assert not sent and not joints


def test_new_run_deadband_does_not_reuse_old_binary_target(monkeypatch):
    r, sent, _, _ = policy_robot()
    r.last_grip = 84.
    monkeypatch.setitem(__import__("sys").modules, "xarm_quest_relative",
                        SimpleNamespace(prepare_servo_hold=lambda *args: (None, np.zeros(7))))
    r.start(lambda: True)
    r.action(np.r_[np.zeros(7), .5], lambda: True)
    assert not sent and r.last_grip is None


def test_engine_clears_cached_actions_before_gripper_transition():
    e = bare_engine()
    e.pending = None
    e.worker = None
    e.cameras = {}
    e.next_action = 0.
    stopped = []
    e.robot = SimpleNamespace(
        controlled_action=lambda *args: dict(status="gripper_transition", gripper_target=0.),
        stop=lambda: stopped.append(True),
    )
    e.actions.extend([np.zeros(8)] * 4)
    e.next_action = time.monotonic()
    e.tick()
    assert stopped == [True] and not e.actions
    assert e.gripper_wait["target"] == 0 and e.gripper_wait["submitted"] is None
    event = e.events.get_nowait()
    assert "确认停稳" in event["text"]
    assert e.events.empty()
