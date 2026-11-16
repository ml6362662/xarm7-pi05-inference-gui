"""Pose semantics, motion routing, cancellation and UI without hardware access."""
import hashlib
import json
import time
import queue
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation

from hardware import Cancelled
from model_contract import (JOINT, TCP, TCP_STATE, TCP_ACTION, LABELS, contract_from_config,
                            sdk_pose_to_state, tcp_action_to_sdk, normalize_tcp_actions,
                            orientation_gap_deg, unit_wxyz)
from pi05_policy import read_bundle
from inference_settings import InferenceSettings
from test_control import bare_engine
from test_settings_execution import configured_robot

FRAME = dict(serial="XS1305", axis=7, device_type=7,
             tcp_offset_mm_rad=[0, 0, 172, 0, 0, 0], world_offset_mm_rad=[0] * 6)
CONTRACT = dict(mode=TCP, label=LABELS[TCP], robot_frame=FRAME)


def tcp_robot():
    r, sent, snapshot, writes = configured_robot()
    snapshot["position_mm"] = 84.
    r.last_grip = 84.
    calls = []
    r.arm.axis, r.arm.device_type, r.arm.sn = 7, 7, "XS1305"
    r.arm.tcp_offset, r.arm.world_offset = FRAME["tcp_offset_mm_rad"], [0] * 6
    r.arm.get_position = lambda **kw: (0, r.fk(r.previous).tolist())
    def ik(pose, **kwargs):
        calls.append((pose, kwargs))
        return 0, np.r_[(np.array(pose[:3]) - [0, 0, 500]) / 1000, pose[3:], 0].tolist()
    r.arm.get_inverse_kinematics = ik
    r.configure_policy(CONTRACT)
    return r, calls, writes


def test_round_trip_units_wxyz_and_sdk_orientation():
    # Includes pitch singularity and angles crossing pi.
    for angles in ([.4, -.6, 2.9], [0, np.pi/2, 1.], [np.pi + .02, .1, -.4]):
        pose = np.r_[[210., -320., 450.], angles]
        state = sdk_pose_to_state(pose, .75)
        np.testing.assert_allclose(state[:3], [.21, -.32, .45])
        restored = tcp_action_to_sdk(state)
        np.testing.assert_allclose(restored[:3], pose[:3], atol=2e-5)
        assert (Rotation.from_euler("xyz", restored[3:]).inv()
                * Rotation.from_euler("xyz", angles)).magnitude() < 1e-6
        assert state[7] == .75


def test_measured_quaternion_continuity_across_pi():
    a = sdk_pose_to_state([0, 0, 500, np.pi-.01, 0, 0], 1)
    b = sdk_pose_to_state([0, 0, 500, np.pi+.01, 0, 0], 1, a[3:7])
    assert np.dot(a[3:7], b[3:7]) > .99
    assert b[3] < 0  # Do not independently force w positive every frame.
    assert orientation_gap_deg(a[3:7], b[3:7]) == pytest.approx(np.rad2deg(.02), abs=1e-5)


@pytest.mark.parametrize("q", [[0, 0, 0, 0], [np.nan, 0, 0, 1], [1, 0, 0], [np.inf, 0, 0, 0]])
def test_bad_quaternions_rejected(q):
    with pytest.raises(ValueError): unit_wxyz(q)


def test_action_projection_preserves_position_gripper_and_rotation():
    state = np.r_[[.2, .3, .5], [-1, 0, 0, 0], .9]
    a = np.tile([.2, .3, .4, 2., 0, 0, 0, .1], (50, 1))
    a[1::2, 3] *= -1
    out = normalize_tcp_actions(a, state)
    np.testing.assert_allclose(out[:, :3], a[:, :3])
    np.testing.assert_allclose(out[:, 7], a[:, 7])
    np.testing.assert_allclose(out[:, 3], -1)
    assert np.all(a[::2, 3] == 2)  # No mutation of raw caller data.


def test_pose_routes_to_ik_with_mm_and_measured_joint_seed():
    r, calls, writes = tcp_robot()
    action = np.array([.001, .002, .501, 1, 0, 0, 0, 1.])
    result = r.controlled_action(action, lambda: True, InferenceSettings(diagnostic_constraints=False))
    assert len(writes) == 1 and len(calls) == 1
    np.testing.assert_allclose(calls[0][0], [1, 2, 501, 0, 0, 0])
    assert calls[0][1] == dict(input_is_radian=True, return_is_radian=True,
                              limited=False, ref_angles=[0.] * 7)
    assert result["model_mode"] == TCP
    np.testing.assert_allclose(result["raw"], action)
    np.testing.assert_allclose(result["ik_target_joints"], [.001, .002, .001, 0, 0, 0, 0])
    assert len(result["sent"]) == 7


def test_far_valid_pose_ik_target_moves_with_existing_speed_limit():
    r, calls, writes = tcp_robot()
    settings = InferenceSettings(diagnostic_constraints=False)
    result = r.controlled_action(np.array([.4, 0, .5, 1, 0, 0, 0, 1.]),
                                 lambda: True, settings)
    assert len(writes) == len(calls) == 1
    assert result["ik_target_joints"][0] == pytest.approx(.4)
    assert 0 < result["sent"][0] <= settings.joint_step_speed / settings.control_hz


@pytest.mark.parametrize("error", ["ik", "shape", "residual", "cancel", "quat"])
def test_invalid_or_cancelled_pose_never_sends_motion(error):
    r, calls, writes = tcp_robot()
    action = np.array([.001, 0, .5, 1, 0, 0, 0, 1.])
    allowed = [True]
    if error == "ik": r.arm.get_inverse_kinematics = lambda *a, **k: (1, [])
    if error == "shape": r.arm.get_inverse_kinematics = lambda *a, **k: (0, [0] * 6)
    if error == "residual": r.arm.get_inverse_kinematics = lambda *a, **k: (0, [.1] + [0] * 6)
    if error == "cancel":
        original = r.arm.get_inverse_kinematics
        def cancel(*a, **k):
            allowed[0] = False
            return original(*a, **k)
        r.arm.get_inverse_kinematics = cancel
    if error == "quat": action[3:7] = 0
    with pytest.raises((RuntimeError, ValueError, Cancelled)):
        r.controlled_action(action, lambda: allowed[0], InferenceSettings())
    assert not writes


def test_frame_validation_and_switch_back_to_joint():
    r, calls, writes = tcp_robot()
    r.arm.tcp_offset = [0] * 6
    with pytest.raises(RuntimeError, match="tcp_offset"):
        r.configure_policy(CONTRACT)
    r.configure_policy({"mode": JOINT})
    r.controlled_action(np.r_[[.001] * 7, 1], lambda: True, InferenceSettings())
    assert len(writes) == 1 and not calls


def test_tcp_policy_state_is_measured_pose_not_joint_angles():
    r, _, _ = tcp_robot()
    r.arm.get_position = lambda **k: (0, [200, 300, 400, 0, 0, 0])
    s = r.policy_state()
    np.testing.assert_allclose(s, [.2, .3, .4, 1, 0, 0, 0, 1])
    np.testing.assert_allclose(r.last_policy_joints, np.zeros(7))


def test_tcp_engine_uses_joint_feedback_for_drift_and_pose_for_display():
    e = bare_engine()
    e.model_contract = CONTRACT
    e.pending.update(state=np.array([.2, .3, .5, 1, 0, 0, 0, 1.]), joint_state=np.zeros(7))
    actions = np.tile([.21, .3, .5, 1, 0, 0, 0, 1.], (50, 1))
    e.result(dict(kind="actions", token=2, actions=actions, elapsed=.1))
    assert len(e.actions) == e.settings.execute_frames
    assert e.inference_info["position_delta_mm"] == pytest.approx(10)
    assert "first_joints_deg" not in e.inference_info
    assert e.inference_info["first_tcp_mm"] == [210, 300, 500]


def test_engine_observation_prediction_execution_routes_tcp_end_to_end():
    e = bare_engine()
    e.model_contract = CONTRACT
    e.robot, calls, writes = tcp_robot()
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    stamp = time.monotonic()
    e.cameras = {name: SimpleNamespace(snapshot=lambda: (image, stamp)) for name in ("global", "wrist")}
    e.pending = e.worker = None
    e.next_action = 0
    e.requests = queue.Queue()
    e.prompt = "test"
    e.tick()
    req = e.requests.get_nowait()
    np.testing.assert_allclose(req["state"], [0, 0, .5, 1, 0, 0, 0, 1])
    np.testing.assert_allclose(e.pending["joint_state"], np.zeros(7))
    a = np.tile(req["state"], (50, 1))
    a[:, 0] += .001
    e.result(dict(kind="actions", token=e.generation, actions=a, elapsed=.1))
    e.tick()
    assert calls and writes
    np.testing.assert_allclose(calls[0][0][:3], [1, 0, 500], atol=1e-6)


def test_cancelled_model_load_cannot_replace_active_contract():
    e = bare_engine()
    e.model_contract = {"mode": JOINT}
    e.load_token = 1
    e.result(dict(kind="loaded", token=1, path="ignored", contract=CONTRACT))
    assert not e.loaded and e.model_contract["mode"] == JOINT


def test_bundle_selects_semantics_from_config_and_requires_matching_spec(tmp_path):
    path = tmp_path / "same_dimension.pt"
    path.write_bytes(b"test")
    cfg = {"actor": {"model": {"action_dim": 8, "num_action_chunks": 50,
           "openpi": dict(config_name="pi05_xarm7_tcp_state", discrete_state_input=True, num_images_in_input=2),
           "openpi_data": {"pad_state": False}}}}
    (tmp_path / "training_config.yaml").write_text(yaml.safe_dump(cfg))
    stats = {k: {name: [val] * n for name, val in (("mean", 0), ("std", 1), ("q01", -1), ("q99", 1))}
             for k, n in (("state", 8), ("actions", 32))}
    (tmp_path / "norm_stats.json").write_text(json.dumps({"norm_stats": stats}))
    (tmp_path / "download_verified.json").write_text(json.dumps({"bytes": 4, "sha256": hashlib.sha256(b"test").hexdigest()}))
    spec = dict(state_semantics=TCP_STATE, action_semantics=TCP_ACTION, robot_frame=FRAME,
                quaternion_policy="unit-normalize; temporal hemisphere continuity within each recording window; action hemisphere aligned to measured state")
    (tmp_path / "dataset_spec.json").write_text(json.dumps(spec))
    assert read_bundle(path)[2]["model_contract"]["mode"] == TCP
    spec["action_semantics"] = "relative tcp xyz"
    (tmp_path / "dataset_spec.json").write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="绝对位姿"): read_bundle(path)
    cfg["actor"]["model"]["openpi"]["config_name"] = "pi05_xarm7_state"
    (tmp_path / "training_config.yaml").write_text(yaml.safe_dump(cfg))
    assert read_bundle(path)[2]["model_contract"]["mode"] == JOINT
