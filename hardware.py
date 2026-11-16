"""Camera and arm adapters shared with the existing collection setup."""
from __future__ import annotations

import fcntl
import json
from pathlib import Path
import sys
import threading
import time

import numpy as np
from joint_angles import continuous_equivalent_angles
from model_contract import (JOINT, TCP, EEFDELTA, POSE_MODES, sdk_pose_to_state,
                            tcp_action_to_sdk, orientation_gap_deg, tcp_state_to_eefdelta)

COLLECTION = Path("/home/a/xarm7-vr-lab")
sys.path.insert(0, str(COLLECTION))


class Cancelled(Exception):
    pass


class SegmentComplete(Exception):
    """A configured review/boundary stop, not a controller fault."""
    def __init__(self, message, trace=None):
        super().__init__(message)
        self.trace = trace or {}


class ResourceLocks:
    def __init__(self, root=COLLECTION):
        self.files = []
        try:
            for name in ("xarm_relative.lock", "recorder_samples.lock"):
                f = (root / ".state" / name).open("a+")
                self.files.append(f)
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.close()
            raise RuntimeError("数据采集/遥操程序仍占用设备，请先关闭其后台控制和采集") from None
        except Exception:
            self.close()
            raise

    def close(self):
        for f in self.files:
            f.close()
        self.files.clear()


class Camera:
    def __init__(self, config):
        self.config = config
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.rgb, self.stamp = None, 0.0
        self.error = "相机预热中"
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        import cv2
        c = self.config
        cap = None
        try:
            if not Path(c["path"]).exists():
                raise RuntimeError(f"未找到相机：{c['path']}")
            cap = cv2.VideoCapture(c["path"], cv2.CAP_V4L2)
            if not cap.isOpened():
                raise RuntimeError("相机被占用或无法打开")
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*c.get("fourcc", "MJPG")))
            for prop, value in (
                (cv2.CAP_PROP_FRAME_WIDTH, c["width"]), (cv2.CAP_PROP_FRAME_HEIGHT, c["height"]),
                (cv2.CAP_PROP_FPS, c["fps"]), (cv2.CAP_PROP_BUFFERSIZE, 1),
            ):
                cap.set(prop, value)
            for key, prop in (
                ("auto_exposure", cv2.CAP_PROP_AUTO_EXPOSURE), ("exposure", cv2.CAP_PROP_EXPOSURE),
                ("autofocus", cv2.CAP_PROP_AUTOFOCUS), ("focus", cv2.CAP_PROP_FOCUS),
            ):
                if key in c:
                    cap.set(prop, float(c[key]))
            warmup = time.monotonic()
            while not self.stop.is_set():
                ok, bgr = cap.read()
                stamp = time.monotonic()
                if not ok or bgr.shape[:2] != (c["height"], c["width"]):
                    raise RuntimeError("相机读帧失败或分辨率不符")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                # Exactly the export orientation: wrist 180 degrees, global 0.
                rgb = np.ascontiguousarray(np.rot90(rgb, k=-(c.get("rotation", 0) // 90)))
                with self.lock:
                    self.rgb, self.stamp = rgb, stamp
                    self.error = "" if stamp - warmup >= 1 else "相机预热中"
        except Exception as error:
            with self.lock:
                self.error = str(error)
        finally:
            if cap is not None:
                cap.release()

    def snapshot(self, require_fresh=True):
        with self.lock:
            image, stamp, error = self.rgb, self.stamp, self.error
        if require_fresh and (error or time.monotonic() - stamp > .3):
            raise RuntimeError(f"{self.config['name']}：{error or '图像超过 300ms 未更新'}")
        return image, stamp

    def close(self):
        self.stop.set()
        self.thread.join(timeout=2)
        if self.thread.is_alive():
            raise RuntimeError("相机线程尚未退出；保留设备锁，请关闭程序后重试")


def checked(code, operation):
    if code != 0:
        raise RuntimeError(f"{operation}失败（SDK {code}）")


def binary_gripper_target(fraction, previous, close_fraction=.4, open_fraction=.6):
    """Schmitt trigger in physical aperture space, after action unnormalization.

    None means no policy command yet. A first prediction in the deadband
    leaves the existing grasp untouched; measured aperture is never binarized.
    """
    if not np.isfinite(fraction) or not -.1 <= fraction <= 1.1:
        raise RuntimeError("预测夹爪开度超范围，已暂停")
    if fraction <= close_fraction:
        return 0.
    if fraction >= open_fraction:
        return 84.
    return previous


def bounded_target(desired, actual, previous, velocity, lower, upper, dt=1/14):
    desired, actual = np.asarray(desired), np.asarray(actual)
    if desired.shape != (8,) or not np.isfinite(desired).all():
        raise RuntimeError("预测动作不是 8 维有限值")
    if np.any(desired[:7] < lower) or np.any(desired[:7] > upper):
        raise RuntimeError("预测动作超出关节限位，已暂停")
    # A far-away valid policy/IK goal is not a controller fault. Move toward it
    # through the existing speed/acceleration limiter; retain feedback-loss and
    # actual command-following guards below.
    if not -.1 <= desired[7] <= 1.1:
        raise RuntimeError("预测夹爪开度超范围，已暂停")
    wanted = np.clip((desired[:7] - previous) / dt, -.35, .35)
    velocity = np.clip(wanted, velocity - .75 * dt, velocity + .75 * dt)
    target = previous + velocity * dt
    # Never chase a stale setpoint more than 0.12 rad ahead of actual feedback.
    if np.max(np.abs(target - actual)) > .12:
        raise RuntimeError("关节跟随误差过大，已暂停")
    if np.any(target < lower) or np.any(target > upper):
        raise RuntimeError("限速后动作超出关节限位")
    return target, velocity, float(np.clip(desired[7], 0, 1))


class Robot:
    def __init__(self, config):
        from xarm.wrapper import XArmAPI
        from xarm.core.config.x_config import XCONF
        from quest_gripper_control import GripperWorker
        from quest_joint_recovery import JointRecovery
        from xarm_quest_relative import wait_initial_reports

        self.arm = self.gripper = None
        self.motion_owned = False
        self.last_grip = None
        self.model_contract = {"mode": JOINT}
        self.policy_quaternion = None
        self.settings = json.loads((COLLECTION / "relative_control_settings.json").read_text())
        try:
            self.arm = XArmAPI(config["robot_ip"], is_radian=True)
            self.arm.set_timeout(1.0)
            wait_initial_reports(self.arm)
            if self.arm.axis != 7:
                raise RuntimeError("连接的不是 7 轴机械臂")
            bounds = np.asarray(XCONF.Robot.JOINT_LIMITS[self.arm.axis][self.arm.device_type])
            self.lower, self.upper = bounds[:, 0] + np.deg2rad(1), bounds[:, 1] - np.deg2rad(1)
            self.recovery = JointRecovery(self.arm, bounds[:, 0], bounds[:, 1], COLLECTION / "relative_home.json")
            self.gripper = GripperWorker(lambda: XArmAPI(config["robot_ip"], is_radian=True))
        except Exception:
            self.close()
            raise

    def health(self, moving=False):
        code, errors = self.arm.get_err_warn_code()
        checked(code, "读取错误状态")
        if errors[0] or errors[1]:
            raise RuntimeError(f"机械臂故障/警告 {errors}，请按厂家流程处理")
        code, state = self.arm.get_state()
        checked(code, "读取运行状态")
        # SDK feedback: 1=moving, 2=sleeping/idle. State 2 is normal
        # between commands and after reaching a target; 3/4/5 remain blocked.
        if moving and state not in (0, 1, 2):
            raise RuntimeError(f"机械臂已进入停止/保护状态 {state}")
        if moving and self.arm.mode != 1:
            raise RuntimeError("机械臂不在关节伺服模式")

    def state(self):
        code, q = self.arm.get_servo_angle(is_radian=True, is_real=True)
        checked(code, "读取实际关节")
        q = np.asarray(q[:7], dtype=np.float32)
        grip = self.gripper.snapshot()
        age = grip["position_age_s"]
        if not grip["connected"] or grip["command_fault"] or grip["error"] or age is None or age > .75:
            raise RuntimeError("夹爪实际开度未就绪/过期：" + grip["error"])
        position = float(grip["position_mm"])
        if not 0 <= position <= 84 or q.shape != (7,) or not np.isfinite(q).all():
            raise RuntimeError("机械臂状态读数无效")
        # Dataset training_vectors divides physical G2 aperture by 84 mm.
        return np.r_[q, position / 84].astype(np.float32)

    def configure_policy(self, contract):
        """Read-only frame check. Never silently change tool/world offsets."""
        if contract["mode"] not in (JOINT, *POSE_MODES):
            raise ValueError("不支持的模型控制模式")
        if contract["mode"] in POSE_MODES:
            frame = contract["robot_frame"]
            for name in ("axis", "device_type", "serial"):
                if name == "device_type" and name not in frame:
                    continue  # Older exports omit this optional SDK subtype.
                actual = getattr(self.arm, "sn" if name == "serial" else name, None)
                if actual != frame[name]:
                    raise RuntimeError(f"TCP 模型机械臂 {name} 不匹配：训练 {frame[name]}，当前 {actual}")
            for name in ("tcp_offset", "world_offset"):
                actual = np.asarray(getattr(self.arm, name), dtype=float)
                if (actual.shape != (6,) or not np.isfinite(actual).all()
                        or not np.allclose(actual, frame[name + "_mm_rad"], rtol=0, atol=1e-3)):
                    raise RuntimeError(f"TCP 模型 {name} 与训练坐标系不一致，请核对设备配置")
        self.model_contract = dict(contract)
        self.policy_quaternion = None

    def policy_state(self):
        joints = self.state()
        self.last_policy_joints = joints[:7].copy()
        if getattr(self, "model_contract", {}).get("mode", JOINT) == JOINT:
            return joints
        code, pose = self.arm.get_position(is_radian=True)
        checked(code, "读取实际 TCP 位姿")
        state = sdk_pose_to_state(pose, joints[7], self.policy_quaternion)
        self.policy_quaternion = state[3:7].copy()
        if self.model_contract["mode"] == EEFDELTA:
            return tcp_state_to_eefdelta(state)
        return state

    def start(self, allowed):
        from xarm_quest_relative import prepare_servo_hold
        self.health()
        self.motion_owned = True
        _, self.previous = prepare_servo_hold(self.arm, allowed)
        self.velocity = np.zeros(7)
        grip = self.gripper.snapshot()
        committed = grip.get("committed_target_mm")
        # Reuse a command actually acknowledged by this worker; don't re-open
        # an already-open gripper on every restart and wait again unnecessarily.
        self.last_grip = committed if committed in (0., 84.) else (
            84. if grip.get("position_mm") is not None and grip["position_mm"] >= 82. else None)

    def begin_segment(self, options):
        """Read-only segment origin; call before enabling servo motion."""
        options.validate()
        self.recovery.validate_profile(self.recovery.home or {})
        code, joint_states = self.arm.get_joint_states(is_radian=True, num=2)
        checked(code, "确认静止")
        velocity = np.asarray(joint_states[1], dtype=float)
        if velocity.shape != (7,) or not np.isfinite(velocity).all():
            raise RuntimeError("关节速度反馈无效")
        if np.max(np.abs(velocity)) > np.deg2rad(.5):
            raise RuntimeError("机械臂未静止")
        state = self.state()
        self.origin_q = state[:7].copy()
        self.origin_pose = self.fk(self.origin_q)
        self.path_mm = 0.
        self.floor = options.floor_mm if options.floor_enabled else None
        if options.gripper_mode == "hold":
            self.floor = max(self.origin_pose[2] - .5, self.floor if self.floor is not None else -np.inf)
        return dict(state=state.tolist(), tcp=self.origin_pose.tolist(), floor_mm=self.floor)

    def fk(self, q):
        code, pose = self.arm.get_forward_kinematics(
            np.asarray(q).tolist(), input_is_radian=True, return_is_radian=True)
        checked(code, "读取运动学正解")
        pose = np.asarray(pose, dtype=float)
        if pose.shape != (6,) or not np.isfinite(pose).all():
            raise RuntimeError("运动学正解无效")
        return pose

    def check_envelope(self, q, pose, options, tolerance=0.):
        if not options.segment_enabled and self.floor is None:
            return
        from scipy.spatial.transform import Rotation
        if options.segment_enabled:
            orientation = (Rotation.from_euler("xyz", self.origin_pose[3:]).inv()
                           * Rotation.from_euler("xyz", pose[3:])).magnitude()
            if (np.linalg.norm(pose[:3] - self.origin_pose[:3]) > options.segment_displacement_mm + tolerance
                    or np.max(np.abs(q - self.origin_q)) > np.deg2rad(options.segment_joint_deg + tolerance)
                    or orientation > np.deg2rad(options.segment_orientation_deg + tolerance)):
                raise SegmentComplete("达到本段位移 / 关节 / 姿态边界")
        if self.floor is not None and pose[2] < self.floor - tolerance:
            raise SegmentComplete("达到本段 TCP 最低高度")

    def direct_eef_action(self, action, allowed, options):
        """Send the decoded EEF target without application trajectory shaping.

        OpenPI-style chunk consumption; xArm IK is the hardware adapter.
        Firmware protections, SDK errors and user cancellation remain active.
        """
        if not allowed():
            raise Cancelled()
        self.health(moving=True)
        action = np.asarray(action, dtype=float)
        if action.shape != (8,) or not np.isfinite(action).all():
            raise RuntimeError("TCP 动作必须是8维有限值")
        state = self.state()
        pose = self.fk(state[:7])
        requested = tcp_action_to_sdk(action)
        code, q = self.arm.get_inverse_kinematics(
            requested.tolist(), input_is_radian=True, return_is_radian=True,
            limited=False, ref_angles=state[:7].tolist())
        checked(code, "TCP 目标逆解")
        target = np.asarray(q, dtype=float)
        if target.shape != (7,) or not np.isfinite(target).all():
            raise RuntimeError("TCP 目标逆解无效")
        raw_ik = target.copy()
        reference = np.asarray(getattr(self, "previous", state[:7]), dtype=float)
        target = continuous_equivalent_angles(raw_ik, reference, self.lower, self.upper)
        if np.any(target < self.lower) or np.any(target > self.upper):
            raise RuntimeError("TCP 目标逆解超出机械臂关节范围")
        target_pose = self.fk(target)
        if not allowed():
            raise Cancelled()
        checked(self.arm.set_servo_angle_j(target.tolist(), is_radian=True), "目标关节伺服")
        # Binary command, matching the official DROID example's 0.5 threshold.
        # G2 has a separate command worker; do not stop the arm or discard the chunk.
        desired_grip = 84. if action[7] > .5 else 0.
        if desired_grip != self.last_grip:
            if not allowed():
                raise Cancelled()
            self.gripper.submit(desired_grip, self.settings["gripper_speed_mm_s"],
                                self.settings["gripper_force"], allowed)
            self.last_grip = desired_grip
        self.previous = target.copy()
        mode = self.model_contract["mode"]
        measured = sdk_pose_to_state(pose, state[7], self.policy_quaternion)
        if mode == EEFDELTA:
            measured = tcp_state_to_eefdelta(measured)
        return dict(status="motion", actual=state.tolist(), actual_tcp=pose.tolist(),
                    raw=action.tolist(), raw_tcp=requested.tolist(),
                    model_mode=mode, model_state=measured.tolist(),
                    actual_joints=state[:7].tolist(), ik_target_joints=target.tolist(),
                    ik_raw_joints=raw_ik.tolist(), ik_reference_joints=reference.tolist(),
                    ik_equivalent_turns=np.rint((target-raw_ik)/(2*np.pi)).astype(int).tolist(),
                    sent=target.tolist(), sent_tcp=target_pose.tolist(),
                    gripper_target=desired_grip,
                    joint_gap_deg=float(np.rad2deg(np.abs(target-state[:7])).max()),
                    tcp_gap_mm=float(np.linalg.norm(requested[:3]-pose[:3])),
                    execution_contract="openpi_chunk_direct_eef_v1",
                    raw_semantics="absolute TCP; original model output is prediction.model_actions")

    def controlled_action(self, action, allowed, options):
        """Continuous execution; optional archived probe constraints for replay.

        Gripper transitions return to Engine for a cancellable stationary wait.
        """
        if not allowed():
            raise Cancelled()
        self.health(moving=True)
        state = self.state()
        actual = state[:7]
        mode = getattr(self, "model_contract", {}).get("mode", JOINT)
        tcp_mode = mode in POSE_MODES
        model_action = np.asarray(action, dtype=float).copy()
        requested_pose = None
        if tcp_mode:
            requested_pose = tcp_action_to_sdk(model_action)
            # Same IK branch selection as collection; use measured joints as seed.
            code, q = self.arm.get_inverse_kinematics(
                requested_pose.tolist(), input_is_radian=True, return_is_radian=True,
                limited=False, ref_angles=actual.tolist())
            checked(code, "TCP 模型目标逆解")
            q = np.asarray(q, dtype=float)
            if q.shape != (7,) or not np.isfinite(q).all():
                raise RuntimeError("TCP 目标逆解不是 7 个有限关节角")
            if not allowed():
                raise Cancelled()
            action = np.r_[q, model_action[7]]
        dt = 1 / options.control_hz
        target, _, fraction = bounded_target(
            action, actual, self.previous, self.velocity, self.lower, self.upper, dt=dt)
        pose = self.fk(actual)
        raw_pose = self.fk(action[:7])
        if tcp_mode:
            solved = sdk_pose_to_state(raw_pose, 0.)
            if (np.linalg.norm(raw_pose[:3] - requested_pose[:3]) > 1.
                    or orientation_gap_deg(solved[3:7], model_action[3:7]) > .5):
                raise RuntimeError("TCP 目标逆解正解不一致，请核对工具与坐标系")
            raw_pose = requested_pose
        trace = dict(actual=state.tolist(), actual_tcp=pose.tolist(),
                     raw=np.asarray(action).tolist(), raw_tcp=raw_pose.tolist(),
                     joint_gap_deg=float(np.rad2deg(np.abs(action[:7] - actual)).max()),
                     tcp_gap_mm=float(np.linalg.norm(raw_pose[:3] - pose[:3])))
        if tcp_mode:
            measured = sdk_pose_to_state(pose, state[7], self.policy_quaternion)
            if mode == EEFDELTA:
                measured = tcp_state_to_eefdelta(measured)
            trace.update(model_mode=mode, raw=model_action.tolist(),
                         model_state=measured.tolist(),
                         actual_joints=actual.tolist(), ik_target_joints=np.asarray(action[:7]).tolist())
            if mode == EEFDELTA:
                trace["raw_semantics"] = "decoded absolute TCP XYZ meters + wxyz + gripper; physical delta is in prediction.model_actions"
        try:
            self.check_envelope(actual, pose, options, tolerance=.5)
            if options.gripper_mode == "review" and fraction < options.review_fraction:
                raise SegmentComplete("模型开始收拢夹爪：已暂停，请确认对齐后手动闭合")
            if options.gripper_mode == "hold" and fraction > max(.2, float(state[7]) + .15):
                raise SegmentComplete("保持夹持期间模型要求打开：已暂停，未发送打开指令")
            desired_grip = binary_gripper_target(
                fraction, self.last_grip, options.close_fraction, options.open_fraction)
            waiting_pose = False
            if options.gripper_mode == "auto" and desired_grip is not None and desired_grip != self.last_grip:
                waiting_pose = (options.diagnostic_constraints and desired_grip == 0. and (
                    trace["tcp_gap_mm"] > options.close_tolerance_mm
                    or trace["joint_gap_deg"] > options.close_tolerance_deg))
                if not waiting_pose:
                    # Do not advance the arm along a chunk while changing grip.
                    return dict(trace, status="gripper_transition", gripper_target=desired_grip)
            delta = np.clip(target - self.previous, -options.joint_step_speed * dt,
                            options.joint_step_speed * dt)
            previous_pose = self.fk(self.previous)
            for _ in range(12):
                target = self.previous + delta
                target_pose = self.fk(target)
                step_mm = float(np.linalg.norm(target_pose[:3] - previous_pose[:3]))
                if not options.diagnostic_constraints or step_mm <= options.tcp_step_mm:
                    break
                delta *= min(.8, options.tcp_step_mm / step_mm)
            else:
                raise RuntimeError("无法满足单步 TCP 位移限制")
            self.check_envelope(target, target_pose, options)
            if options.segment_enabled and self.path_mm + step_mm > options.segment_path_mm:
                raise SegmentComplete("达到本段路径长度上限")
            if options.diagnostic_constraints and np.linalg.norm(target_pose[:3] - pose[:3]) > options.tracking_mm:
                raise RuntimeError("末端实际跟随距离超过设置上限")
            if not allowed():
                raise Cancelled()
            checked(self.arm.set_servo_angle_j(target.tolist(), is_radian=True,
                    speed=options.joint_step_speed, mvacc=options.servo_acceleration), "受控关节伺服")
            self.previous, self.velocity = target, delta / dt
            self.path_mm += step_mm
            return dict(trace, status="await_close_pose" if waiting_pose else "motion",
                        sent=target.tolist(), sent_tcp=target_pose.tolist(), path_mm=self.path_mm)
        except SegmentComplete as stop:
            stop.trace = trace
            raise

    def action(self, action, allowed):
        """Legacy adapter for offline regressions; GUI uses controlled_action."""
        if not allowed():
            raise Cancelled()
        self.health(moving=True)
        actual = self.state()[:7]
        target, velocity, fraction = bounded_target(
            action, actual, self.previous, self.velocity, self.lower, self.upper)
        if not allowed():
            raise Cancelled()
        checked(self.arm.set_servo_angle_j(target.tolist(), is_radian=True, speed=.35, mvacc=.75), "关节伺服")
        self.previous, self.velocity = target, velocity
        aperture = binary_gripper_target(fraction, self.last_grip)
        if not allowed():
            raise Cancelled()
        if aperture is not None and aperture != self.last_grip:
            self.gripper.submit(
                aperture, self.settings["gripper_speed_mm_s"], self.settings["gripper_force"], allowed)
            self.last_grip = aperture

    def manual_gripper(self, aperture, allowed):
        """Use the existing G2 I/O worker; never enable or move arm joints."""
        if not allowed():
            raise Cancelled()
        if type(aperture) not in (int, float) or aperture not in (0., 84.):
            raise ValueError("夹爪仅支持打开（84 mm）或闭合（0 mm）")
        grip = self.gripper.snapshot()
        if (not grip["connected"] or grip["error"] or grip["command_fault"]
                or grip["position_age_s"] is None or grip["position_age_s"] > .75):
            raise RuntimeError("夹爪未就绪或实际开度过期：" + grip["error"])
        if not allowed():
            raise Cancelled()
        self.gripper.submit(
            float(aperture), self.settings["gripper_speed_mm_s"],
            self.settings["gripper_force"], allowed)
        self.last_grip = None  # A later policy run starts a new gripper command sequence.

    def stop(self):
        if self.gripper is not None:
            self.gripper.cancel_pending()
        if self.arm is not None and self.motion_owned:
            # Stop clears buffered goals; resuming always reinitializes from measured pose.
            checked(self.arm.set_state(4), "停止机械臂")
            self.motion_owned = False

    def home(self, allowed, speed_deg_s=5.):
        from quest_control_limits import home_speed
        speed_deg_s = home_speed(speed_deg_s)
        def check_cancel():
            if not allowed():
                raise Cancelled()

        self.stop()
        check_cancel()
        self.recovery.validate_profile(self.recovery.home or {})
        code, q = self.arm.get_servo_angle(is_radian=True, is_real=True)
        checked(code, "读取复位起点")
        plan = self.recovery.plan_home_pose(q, check_cancel, speed_deg_s=speed_deg_s)
        check_cancel()
        self.motion_owned = True
        for operation in (
            lambda: self.arm.set_state(4), lambda: self.arm.set_mode(0),
            lambda: self.arm.motion_enable(enable=True), lambda: self.arm.set_state(0),
        ):
            check_cancel()
            checked(operation(), "准备初始位置复位")
        check_cancel()
        checked(self.arm.set_servo_angle(
            angle=q[:7], is_radian=True, speed=.1, mvacc=.3,
            wait=True, timeout=2), "初始化复位保持目标")
        check_cancel()
        checked(self.arm.set_servo_angle(
            angle=plan["target"], is_radian=True, speed=np.deg2rad(speed_deg_s),
            mvacc=np.deg2rad(2 * speed_deg_s), wait=False), "回到初始位置")
        deadline = time.monotonic() + plan["max_joint_travel_deg"] / speed_deg_s + 15
        settled = 0
        while time.monotonic() < deadline:
            check_cancel()
            self.health()
            code, robot_state = self.arm.get_state()
            checked(code, "核对复位运行状态")
            if robot_state not in (0, 1, 2):
                raise RuntimeError(f"复位被停止，机器人 state={robot_state}")
            code, q = self.arm.get_servo_angle(is_radian=True, is_real=True)
            checked(code, "读取复位进度")
            settled = settled + 1 if np.max(np.abs(np.asarray(q[:7]) - plan["target"])) < np.deg2rad(.5) else 0
            if settled >= 5:
                self.stop()
                return
            time.sleep(.04)
        raise RuntimeError("返回初始位置超时")

    def close(self):
        try:
            self.stop()
        finally:
            if self.gripper is not None:
                self.gripper.close()
            if self.arm is not None:
                self.arm.disconnect()
