"""Explicit training semantics; equal tensor dimensions do not imply equal actions."""
from pathlib import Path
import json
import warnings

import numpy as np
from scipy.spatial.transform import Rotation

JOINT = "joint_absolute"
TCP = "tcp_absolute_wxyz"
EEFDELTA = "tcp_fixed_anchor_delta"
POSE_MODES = (TCP, EEFDELTA)
LABELS = {JOINT: "绝对关节目标", TCP: "绝对末端 Pose（XYZ＋wxyz）",
          EEFDELTA: "末端增量 EEFDelta（固定观测锚点）"}
TCP_STATE = "measured base-frame TCP XYZ meters + unit quaternion wxyz + measured gripper open fraction"
TCP_ACTION = "absolute base-frame TCP XYZ meters + unit quaternion wxyz + absolute gripper open fraction"
QUATERNION_POLICY = "unit-normalize; temporal hemisphere continuity within each recording window; action hemisphere aligned to measured state"
DELTA_STATE = "measured absolute base XYZ meters + rotation first two columns concatenated rot6d + measured gripper open fraction; no joints"
DELTA_ACTION = "each of 50 commanded targets relative to ONE measured chunk anchor: base XYZ delta meters + Log(R_target @ R_anchor.T) rotation vector radians + absolute commanded gripper open fraction"
DELTA_ROTATION = "R maps TCP to base; reconstruct Exp(rotvec) @ R_anchor; rot6d columns with Gram-Schmidt"


def state_dim(mode):
    return 10 if mode == EEFDELTA else 8


def action_dim(mode):
    return 7 if mode == EEFDELTA else 8


def contract_from_config(config, model_path):
    name = config["actor"]["model"]["openpi"]["config_name"]
    if name == "pi05_xarm7_state":
        return dict(mode=JOINT, label=LABELS[JOINT], state="joint radians + gripper fraction",
                    action="absolute joint radians + gripper fraction")
    if name not in ("pi05_xarm7_tcp_state", "pi05_xarm7_eefdelta_state"):
        raise ValueError("不支持的模型动作定义：" + str(name))
    # Sidecar travels with the downloaded training snapshot, not a filename guess.
    folder = Path(model_path).parent
    spec_path = folder / "dataset_spec.json"
    if not spec_path.is_file():
        spec_path = folder.parent / "training_snapshot/dataset_spec.json"
    spec = json.loads(spec_path.read_text())
    mode = EEFDELTA if name == "pi05_xarm7_eefdelta_state" else TCP
    expected_state, expected_action = (DELTA_STATE, DELTA_ACTION) if mode == EEFDELTA else (TCP_STATE, TCP_ACTION)
    if spec.get("state_semantics") != expected_state or spec.get("action_semantics") != expected_action:
        if mode == EEFDELTA:
            raise ValueError("EEFDelta 状态／动作语义不匹配，必须是10维实测状态和7维固定锚点基座增量")
        raise ValueError("TCP 模型必须为基座坐标、米、wxyz 四元数的绝对位姿；不支持自动猜测增量定义")
    if mode == EEFDELTA:
        if (spec.get("state_dim"), spec.get("action_dim"), spec.get("model_action_dim"), spec.get("horizon")) != (10, 7, 32, 50):
            raise ValueError("EEFDelta 模型必须是状态10维、动作7维、网络32维、50帧")
        if spec.get("rotation_convention") != DELTA_ROTATION:
            raise ValueError("EEFDelta 旋转定义必须是基座增量左乘固定锚点")
    if spec.get("quaternion_policy") != QUATERNION_POLICY:
        raise ValueError("TCP 模型四元数符号约定与当前适配器不一致")
    frame = spec["robot_frame"]
    if frame.get("axis") != 7:
        raise ValueError("TCP 模型不是 7 轴机械臂配置")
    for key in ("tcp_offset_mm_rad", "world_offset_mm_rad"):
        a = np.asarray(frame[key], dtype=float)
        if a.shape != (6,) or not np.isfinite(a).all():
            raise ValueError("TCP 模型坐标系配置无效")
    return dict(mode=mode, label=LABELS[mode], state=expected_state, action=expected_action,
                state_dim=state_dim(mode), action_dim=action_dim(mode),
                control_action="absolute base XYZ meters + unit wxyz + absolute gripper fraction",
                robot_frame=frame, quaternion_policy=spec["quaternion_policy"],
                **({"training_fps": spec["training_fps"]} if "training_fps" in spec else {}),
                **({"rotation_convention": DELTA_ROTATION} if mode == EEFDELTA else {}))


def unit_wxyz(value, reference=None):
    q = np.asarray(value, dtype=float)
    norm = np.linalg.norm(q)
    if q.shape != (4,) or not np.isfinite(q).all() or not np.isfinite(norm) or norm < 1e-8:
        raise ValueError("TCP 四元数必须是 4 个有限值且模长非零")
    q = q / norm
    if (reference is None and q[0] < 0) or (reference is not None and np.dot(q, reference) < 0):
        q = -q
    return q


def sdk_pose_to_state(pose, aperture_fraction, reference=None):
    """xArm mm + SDK RPY -> training m + continuous wxyz."""
    pose = np.asarray(pose, dtype=float)
    if pose.shape != (6,) or not np.isfinite(pose).all() or not np.isfinite(aperture_fraction):
        raise ValueError("实测 TCP 位姿无效")
    q = Rotation.from_euler("xyz", pose[3:]).as_quat()[[3, 0, 1, 2]]
    return np.r_[pose[:3] / 1000, unit_wxyz(q, reference), aperture_fraction].astype(np.float32)


def tcp_action_to_sdk(action):
    """Training absolute meters + wxyz -> xArm absolute mm + RPY radians."""
    action = np.asarray(action, dtype=float)
    if action.shape != (8,) or not np.isfinite(action).all():
        raise ValueError("TCP 动作必须是 8 个有限值")
    q = unit_wxyz(action[3:7])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Gimbal lock detected.*", category=UserWarning)
        rpy = Rotation.from_quat(q[[1, 2, 3, 0]]).as_euler("xyz")
    return np.r_[action[:3] * 1000, rpy]


def normalize_tcp_actions(actions, state):
    a = np.asarray(actions, dtype=np.float32).copy()
    if a.ndim != 2 or a.shape[1] != 8 or not np.isfinite(a).all():
        raise ValueError("TCP 动作形状无效或含 NaN/Inf")
    reference = unit_wxyz(np.asarray(state)[3:7], np.asarray(state)[3:7])
    for row in a:
        row[3:7] = unit_wxyz(row[3:7], reference)
        reference = row[3:7]
    return a


def orientation_gap_deg(a, b):
    qa, qb = unit_wxyz(a), unit_wxyz(b)
    return float(np.rad2deg(2 * np.arccos(np.clip(abs(np.dot(qa, qb)), 0, 1))))


def tcp_state_to_eefdelta(state):
    """Measured absolute XYZ/wxyz/grip -> XYZ + first two rotation columns + grip."""
    s = np.asarray(state, dtype=float)
    if s.shape != (8,) or not np.isfinite(s).all():
        raise ValueError("实测 TCP 状态必须是8个有限值")
    q = unit_wxyz(s[3:7])
    r = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
    return np.r_[s[:3], r[:, 0], r[:, 1], s[7]].astype(np.float32)


def eefdelta_state_to_tcp(state):
    """Recover a fixed measured chunk anchor with the training Gram-Schmidt rule."""
    s = np.asarray(state, dtype=float)
    if s.shape != (10,) or not np.isfinite(s).all():
        raise ValueError("EEFDelta 状态必须是10个有限值")
    x, y = s[3:6].copy(), s[6:9].copy()
    nx = np.linalg.norm(x)
    if nx < 1e-8:
        raise ValueError("EEFDelta rot6d 第一列退化")
    x /= nx
    y -= np.dot(x, y) * x
    ny = np.linalg.norm(y)
    if ny < 1e-8:
        raise ValueError("EEFDelta rot6d 第二列退化")
    y /= ny
    r = np.stack([x, y, np.cross(x, y)], axis=-1)
    q = Rotation.from_matrix(r).as_quat()[[3, 0, 1, 2]]
    return np.r_[s[:3], unit_wxyz(q), s[9]]


def decode_eefdelta_actions(state, actions):
    """Decode physical actions once per chunk, never relative to the previous target."""
    anchor = eefdelta_state_to_tcp(state)
    a = np.asarray(actions, dtype=float)
    if a.ndim != 2 or a.shape[1] != 7 or a.shape[0] == 0 or not np.isfinite(a).all():
        raise ValueError("EEFDelta 动作必须是非空的 N×7 有限数组")
    rotation = Rotation.from_quat(anchor[[4, 5, 6, 3]])
    targets = Rotation.from_rotvec(a[:, 3:6]) * rotation
    q = targets.as_quat()[:, [3, 0, 1, 2]]
    absolute = np.c_[anchor[:3] + a[:, :3], q, a[:, 6]]
    return normalize_tcp_actions(absolute, anchor)
