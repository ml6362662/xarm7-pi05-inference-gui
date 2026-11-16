"""Cancelable inference/control state machine. The GPU worker never owns hardware."""
from __future__ import annotations

from collections import deque
import json
import multiprocessing as mp
from pathlib import Path
import queue
import threading
import time
import traceback

from hardware import COLLECTION, Camera, Cancelled, ResourceLocks, Robot, SegmentComplete
from pi05_policy import policy_worker
from inference_settings import InferenceSettings, describe, simple_settings
from run_recording import RunRecorder
from model_contract import (JOINT, TCP, EEFDELTA, POSE_MODES, LABELS, action_dim,
                            tcp_action_to_sdk, orientation_gap_deg,
                            decode_eefdelta_actions, eefdelta_state_to_tcp)


class Engine:
    def __init__(self):
        self.events = queue.Queue()
        self.commands = queue.Queue()
        self.cancel = threading.Event()
        self.cancel.set()
        self.closing = threading.Event()
        self.closed = threading.Event()
        self.heartbeat = time.monotonic()
        self.generation = 0
        self.robot = self.locks = None
        self.cameras = {}
        self.loaded = False
        self.loading = False
        self.running = False
        self.pending = None
        self.actions = deque()
        self.worker = None
        self.prompt = ""
        self.next_action = 0.
        self.load_token = None
        self.connected = False
        self.settings = InferenceSettings()
        self.model_path = None
        self.model_verification = None
        self.model_contract = {"mode": JOINT, "label": LABELS[JOINT]}
        self.recorder = None
        self.cancel_reason = ""
        self.gripper_wait = None
        self.segment_deadline = float("inf")
        self.segment_predictions = 0
        self.last_control_tick = time.monotonic()
        self.last_image_record = -float("inf")
        # Immutable small CPU snapshots, overwritten instead of queued.
        self.inference_info = None
        self.inference_count = 0
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()
        self.watchdog = threading.Thread(target=self.supervise, daemon=True)
        self.watchdog.start()

    def emit(self, text, **fields):
        fields.setdefault("gripper_waiting", self.gripper_wait is not None)
        self.events.put(dict(text=text, loaded=self.loaded, loading=self.loading,
                             connected=self.connected, running=self.running, **fields))

    def record(self, kind, data, images=None):
        if self.recorder:
            self.recorder.record(kind, data, images)

    def supervise(self):
        while not self.closing.wait(.04):
            robot = self.robot
            if self.running and robot and robot.motion_owned:
                if not self.allowed(self.generation) or time.monotonic() - self.last_control_tick > .8:
                    self.cancel.set()
                    try:
                        robot.stop()
                        self.emit("控制心跳超时：独立监控已停止机械臂")
                    except Exception as error:
                        self.emit(f"独立停止失败：{error}；请使用实体急停", critical=True)

    def allowed(self, token):
        return (token == self.generation and not self.cancel.is_set() and
                not self.closing.is_set() and time.monotonic() - self.heartbeat < 1.2)

    def request(self, kind, **data):
        if kind in ("pause", "home", "load", "shutdown", "gripper", "start"):
            self.cancel_reason = {
                "pause": "用户暂停", "home": "回到初始位置", "load": "重新加载模型",
                "shutdown": "退出程序", "gripper": "手动控制夹爪", "start": "重新开始推理",
            }[kind]
            self.generation += 1
            self.cancel.set()
        token = self.generation
        if kind == "shutdown":
            self.closing.set()
        self.commands.put(dict(kind=kind, token=token, **data))

    def start_worker(self):
        if self.worker is None or not self.worker.is_alive():
            ctx = mp.get_context("spawn")
            self.requests, self.replies = ctx.Queue(maxsize=2), ctx.Queue()
            self.worker = ctx.Process(target=policy_worker, args=(self.requests, self.replies), daemon=True)
            self.worker.start()

    def halt(self, reason=None):
        self.cancel.set()
        self.running = False
        self.actions.clear()
        self.gripper_wait = None
        try:
            if self.robot:
                self.robot.stop()
        finally:
            if self.recorder:
                recorder = self.recorder
                reason = reason or getattr(self, "cancel_reason", "") or "已停止"
                self.record("stop", dict(reason=reason))
                self.recorder = None
                try:
                    warning = recorder.finish(reason)
                except Exception as error:
                    warning = "结束记录失败：" + str(error)
                self.emit("本次记录已结束，等待标注成功或失败：" + str(recorder.path),
                          completed_recording=str(recorder.path), recording_warning=warning)
                if warning:
                    self.emit("运行记录提示：" + warning)

    def connect(self):
        if self.connected:
            return
        self.locks = ResourceLocks()
        try:
            config = json.loads((COLLECTION / "config.json").read_text())
            configs = {c["name"]: c for c in config["cameras"]}
            if not {"global", "wrist"} <= configs.keys():
                raise RuntimeError("采集配置缺少 global / wrist 相机")
            self.robot = Robot(config)
            self.cameras = {name: Camera(configs[name]) for name in ("global", "wrist")}
            self.connected = True
            self.emit("设备已连接，等待相机和夹爪状态就绪")
        except Exception:
            self.disconnect()
            raise

    def disconnect(self):
        for camera in self.cameras.values():
            camera.close()
        self.cameras.clear()
        if self.robot:
            self.robot.close()
            self.robot = None
        if self.locks:
            self.locks.close()
            self.locks = None
        self.connected = False

    def observations(self):
        global_image, a = self.cameras["global"].snapshot()
        wrist_image, b = self.cameras["wrist"].snapshot()
        if abs(a - b) > .1:
            raise RuntimeError("双相机图像时间差超过 100ms")
        if getattr(self, "model_contract", {}).get("mode", JOINT) in POSE_MODES:
            state = self.robot.policy_state()
            self.observation_joints = self.robot.last_policy_joints.copy()
        else:
            state = self.robot.state()
            self.observation_joints = state[:7].copy()
        return global_image, wrist_image, state

    def command(self, req):
        kind, token = req["kind"], req["token"]
        if kind == "pause":
            self.halt()
            self.emit("已暂停；未执行动作已清空，夹爪保持当前抓持目标")
        elif kind == "connect":
            self.connect()
        elif kind == "load":
            self.halt()
            if self.loading or self.pending is not None:
                raise RuntimeError("上一条模型任务尚未完成，请稍后加载")
            self.loaded = False
            self.loading = True
            self.inference_info = None
            self.inference_count = 0
            self.load_token = token
            self.start_worker()
            self.requests.put_nowait(dict(kind="load", token=token, path=req["path"]))
            self.emit("正在加载模型…")
        elif kind == "start":
            if token != self.generation:
                return
            if not self.loaded or self.loading or not self.connected:
                raise RuntimeError("请先加载模型并连接设备")
            if self.running:
                return
            if self.pending is not None:
                raise RuntimeError("上一次推理正在退出，请稍后重新开始")
            options = InferenceSettings.from_dict(req.get("settings", {}))
            record_run = req.get("record_run", True)
            if type(record_run) is not bool:
                raise ValueError("记录本次推理必须是勾选状态")
            if hasattr(self.robot, "configure_policy"):
                self.robot.configure_policy(getattr(self, "model_contract", {"mode": JOINT}))
            self.observations()
            self.prompt = req["prompt"].strip()
            if not self.prompt:
                raise RuntimeError("任务描述不能为空")
            if self.gripper_wait is not None:
                raise RuntimeError("夹爪开合等待尚未完成")
            if self.model_contract["mode"] in POSE_MODES:
                from dataclasses import replace
                options = replace(simple_settings(options), control_hz=30, gripper_wait_s=0.)
            origin = self.robot.begin_segment(options)
            self.settings = options
            self.recorder = RunRecorder(dict(
                settings=options.to_dict(), model_path=self.model_path,
                model_verification=self.model_verification,
                model_contract=getattr(self, "model_contract", {"mode": JOINT}),
                prompt=self.prompt, origin=origin,
                execution_contract=("sequential_direct_pose_30hz_v1" if self.model_contract["mode"] in POSE_MODES else "legacy"))) if record_run else None
            self.cancel_reason = ""
            self.cancel.clear()
            if not self.allowed(token):
                raise Cancelled()
            self.robot.start(lambda: self.allowed(token))
            if not self.allowed(token):
                raise Cancelled()
            self.running = True
            self.last_control_tick = time.monotonic()
            self.segment_deadline = (time.monotonic() + options.segment_seconds
                                     if options.segment_enabled else float("inf"))
            self.segment_predictions = 0
            self.last_image_record = -float("inf")
            self.actions.clear()
            self.next_action = time.monotonic()
            self.emit("推理运行中：" + describe(options))
            self.emit("本轮详细记录：" + str(self.recorder.path) if self.recorder else "本次不保存推理轨迹和图片",
                      active_recording=str(self.recorder.path) if self.recorder else "")
        elif kind == "home":
            self.halt()
            if token != self.generation:
                return
            if not self.connected:
                raise RuntimeError("请先连接设备")
            self.cancel.clear()
            options = InferenceSettings.from_dict(req.get("settings", self.settings.to_dict()))
            speed = options.home_speed_deg_s
            self.emit(f"正在回到采集程序保存的初始位置（{speed:g}°/s）…", homing=True)
            self.robot.home(lambda: self.allowed(token), speed_deg_s=speed)
            self.cancel.set()
            self.emit("已回到初始位置；推理保持暂停，夹爪开度未改变", homing=False)
        elif kind == "gripper":
            self.halt()
            if token != self.generation:
                return
            if not self.connected:
                raise RuntimeError("请先连接设备")
            self.settings = InferenceSettings.from_dict(req.get("settings", self.settings.to_dict()))
            self.cancel.clear()
            submitted = time.monotonic()
            self.robot.manual_gripper(
                req["aperture"], lambda: self.allowed(token) and not self.running)
            self.gripper_wait = dict(target=req["aperture"], submitted=submitted,
                                     started=submitted, acknowledged=None, resume=False, token=token)
            self.emit(f"推理已暂停；已提交夹爪开度 {req['aperture']:.1f} mm，"
                      f"下发后等待 {self.settings.gripper_wait_s:g} 秒", homing=False)

    def result(self, reply):
        kind, token = reply["kind"], reply["token"]
        if kind == "progress":
            self.emit(reply["text"])
        elif kind == "loaded":
            if token != self.load_token or token != self.generation:
                self.loading = False
                self.loaded = False
                self.emit("模型加载结果已取消，请重新加载")
                return
            contract = reply.get("contract", {})
            if contract.get("mode") not in (JOINT, *POSE_MODES):
                self.loading = self.loaded = False
                raise RuntimeError("模型未返回受支持的状态／动作定义，拒绝启用")
            self.loading = False
            self.loaded = True
            self.model_path = reply["path"]
            self.model_verification = reply.get("verification")
            self.model_contract = contract
            self.emit("模型已加载并完成 GPU 预热 · " + self.model_contract["label"],
                      model_path=reply["path"], model_mode=self.model_contract["mode"],
                      model_label=self.model_contract["label"])
        elif kind == "error":
            self.pending = None
            if self.loading:
                self.loading = False
                self.loaded = False
            self.halt("模型错误：" + reply["text"])
            self.emit("错误：" + reply["text"], detail=reply.get("trace", ""))
        elif kind == "actions":
            pending, self.pending = self.pending, None
            if pending is None or token != pending["token"] or not self.allowed(token) or not self.running:
                return  # A canceled result must never replenish the command queue.
            import numpy as np
            if time.monotonic() - pending["started"] > 3:
                raise RuntimeError("推理结果超过 3 秒，已丢弃过期动作")
            state = self.robot.state() if self.model_contract["mode"] not in POSE_MODES else None
            if self.model_contract["mode"] not in POSE_MODES and np.max(np.abs(state[:7] - pending.get("joint_state", pending["state"][:7]))) > np.deg2rad(self.settings.inference_drift_deg):
                raise RuntimeError("推理期间关节位置已变化，请重新开始")
            mode = getattr(self, "model_contract", {}).get("mode", JOINT)
            raw_actions = np.asarray(reply["actions"])
            if raw_actions.shape != (50, action_dim(mode)) or not np.isfinite(raw_actions).all():
                raise RuntimeError("模型动作形状/数值无效")
            # The pending state is the immutable measured anchor for ALL 50 targets.
            # Decode before queuing. Neither each tick nor gripper resume reanchors a chunk.
            a = decode_eefdelta_actions(pending["state"], raw_actions) if mode == EEFDELTA else raw_actions
            # OpenPI-style configurable open-loop horizon; no hidden model-specific cap.
            execute_frames = self.settings.execute_frames
            self.actions.extend(a[:execute_frames])
            self.current_action_index = 0
            self.next_action = time.monotonic()
            self.inference_count += 1
            self.inference_info = {
                "sequence": self.inference_count,
                "received_at": time.time(), "monotonic": time.monotonic(),
                "elapsed_ms": float(reply["elapsed"] * 1000),
                "shape": tuple(raw_actions.shape), "execute_frames": execute_frames,
                "first_joints_deg": np.rad2deg(a[0, :7]).tolist(),
                "first_gripper_mm": float(a[0, 7] * 84),
                "max_joint_delta_deg": float(np.rad2deg(np.abs(a[0, :7] - pending["state"][:7])).max()),
            }
            if mode in POSE_MODES:
                pose = tcp_action_to_sdk(a[0])
                anchor = eefdelta_state_to_tcp(pending["state"]) if mode == EEFDELTA else pending["state"]
                self.inference_info.pop("first_joints_deg")
                self.inference_info.pop("max_joint_delta_deg")
                self.inference_info.update(model_mode=mode, first_tcp_mm=pose[:3].tolist(),
                    first_rpy_deg=np.rad2deg(pose[3:]).tolist(), first_quaternion_wxyz=a[0, 3:7].tolist(),
                    position_delta_mm=float(np.linalg.norm(a[0, :3] - anchor[:3]) * 1000),
                    orientation_delta_deg=orientation_gap_deg(a[0, 3:7], anchor[3:7]))
                if mode == EEFDELTA:
                    self.inference_info.update(first_delta_xyz_mm=(raw_actions[0, :3] * 1000).tolist(),
                        first_rotvec_deg=np.rad2deg(raw_actions[0, 3:6]).tolist())
            self.record("prediction", dict(sequence=self.inference_count,
                        state=pending["state"].tolist(), actions=a.tolist(),
                        **({"model_mode": mode, "model_actions": raw_actions.tolist(),
                            "anchor_state": pending["state"].tolist(),
                            "actions_semantics": "decoded absolute TCP XYZ meters + wxyz + gripper"}
                           if mode in POSE_MODES else {}),
                        elapsed_ms=reply["elapsed"] * 1000))

    def tick_gripper_wait(self):
        wait = self.gripper_wait
        if not self.allowed(wait["token"]):
            raise Cancelled()
        self.robot.health()
        now = time.monotonic()
        if wait["submitted"] is None:
            import numpy as np
            from hardware import checked
            code, states = self.robot.arm.get_joint_states(is_radian=True, num=2)
            checked(code, "确认开合前机械臂停稳")
            velocity = np.asarray(states[1], dtype=float)
            if velocity.shape != (7,) or not np.isfinite(velocity).all():
                raise RuntimeError("停稳检查的关节速度无效")
            if np.max(np.abs(velocity)) > np.deg2rad(.5):
                if now - wait["started"] > 2:
                    raise RuntimeError("机械臂未在 2 秒内停稳，未发送夹爪动作")
                return
            wait["submitted"] = now
            self.robot.manual_gripper(wait["target"], lambda: self.allowed(wait["token"]))
            self.robot.last_grip = wait["target"]
            self.record("gripper_submit", dict(target=wait["target"]))
        grip = self.robot.gripper.snapshot()
        if (not grip["connected"] or grip["command_fault"] or grip["error"]
                or grip["position_age_s"] is None or grip["position_age_s"] > .75):
            raise RuntimeError("夹爪等待期间反馈失效或命令失败")
        ack = grip.get("command_time_s")
        if (wait["acknowledged"] is None and ack is not None and ack >= wait["submitted"]
                and grip.get("committed_target_mm") == wait["target"]):
            wait["acknowledged"] = ack
            self.record("gripper_ack", dict(target=wait["target"], command_time_s=ack))
        if wait["acknowledged"] is None:
            if now - wait["submitted"] > 1.5:
                raise RuntimeError("夹爪命令未在 1.5 秒内确认下发，已暂停")
            return
        if now - wait["acknowledged"] < self.settings.gripper_wait_s:
            return
        # A time delay is not a claim that an object was successfully grasped.
        self.record("gripper_wait_done", dict(target=wait["target"], actual_mm=grip["position_mm"]))
        self.gripper_wait = None
        if wait["resume"]:
            self.last_control_tick = now
            self.robot.start(lambda: self.allowed(wait["token"]))
            self.robot.last_grip = wait["target"]
            self.actions.clear()
            self.next_action = time.monotonic()
            self.segment_deadline += self.next_action - wait["started"]
            self.emit("夹爪等待结束；重新观察后继续推理（未判断抓取是否成功）")
        else:
            self.cancel.set()
            self.emit("夹爪等待结束；推理保持暂停，请确认实际夹持情况")

    def tick(self):
        self.last_control_tick = time.monotonic()
        if self.worker is not None and not self.worker.is_alive() and (self.loaded or self.loading):
            self.loaded = self.loading = False
            self.pending = None
            raise RuntimeError("模型进程已退出，请重新加载模型")
        if self.worker is not None:
            while True:
                try:
                    self.result(self.replies.get_nowait())
                except queue.Empty:
                    break
        if self.gripper_wait is not None:
            self.tick_gripper_wait()
            return
        if not self.running:
            return
        token = self.generation
        if not self.allowed(token):
            raise Cancelled()
        if time.monotonic() > self.segment_deadline:
            raise SegmentComplete("达到本段时间上限")
        # Check live cameras even while a previous chunk is being executed.
        for camera in self.cameras.values():
            camera.snapshot()
        if self.pending is not None:
            if time.monotonic() - self.pending["started"] > 3:
                raise RuntimeError("模型计算超时，已暂停")
            return
        now = time.monotonic()
        if self.actions:
            if now < self.next_action:
                return
            if self.model_contract["mode"] not in POSE_MODES and now - self.next_action > .2:
                raise RuntimeError("动作控制循环延迟过大，已暂停")
            action = self.actions.popleft()
            execute = (self.robot.direct_eef_action if self.model_contract["mode"] in POSE_MODES
                       else self.robot.controlled_action)
            started = time.monotonic()
            result = execute(action, lambda: self.allowed(token), self.settings)
            self.record("action", dict(sequence=self.inference_count, action_index=self.current_action_index,
                        sent_at=started, dispatch_seconds=time.monotonic()-started, **result))
            self.current_action_index += 1
            if result["status"] == "gripper_transition":
                target = result["gripper_target"]
                self.robot.stop()
                if not self.allowed(token):
                    raise Cancelled()
                self.gripper_wait = dict(target=target, submitted=None, started=time.monotonic(),
                                         acknowledged=None, resume=True, token=token)
                self.actions.clear()
                self.emit(f"已请求停止；确认停稳后夹爪{'打开' if target == 84 else '闭合'}，"
                          f"下发后等待 {self.settings.gripper_wait_s:g} 秒")
            elif result["status"] == "await_close_pose":
                self.actions.clear()  # Never execute cached lifting frames before closing.
            period = 1/self.settings.control_hz
            self.next_action += period
            if self.next_action < time.monotonic():
                # Preserve every prediction row in order; never burst to catch up.
                self.next_action = time.monotonic()
        else:
            if now < self.next_action:
                return
            if self.settings.segment_enabled and self.segment_predictions >= self.settings.segment_cycles:
                raise SegmentComplete("达到本段预测次数上限")
            global_image, wrist_image, state = self.observations()
            self.segment_predictions += 1
            images = None
            if self.recorder and self.settings.save_images and now - self.last_image_record >= self.settings.image_interval_s:
                images = dict(global_rgb=global_image, wrist_rgb=wrist_image)
                self.last_image_record = now
            self.record("observation", dict(sequence=self.inference_count + 1, state=state.tolist()), images)
            self.pending = dict(token=token, state=state,
                                joint_state=getattr(self, "observation_joints", state[:7]).copy(), started=now)
            self.requests.put_nowait(dict(kind="infer", token=token, **{
                "global": global_image, "wrist": wrist_image,
                "state": state, "prompt": self.prompt, "settings": self.settings.to_dict(),
            }))

    def run(self):
        try:
            while not self.closing.is_set():
                try:
                    if self.running and not self.allowed(self.generation):
                        raise Cancelled()
                    try:
                        timeout = .01
                        if self.running and self.pending is None:
                            timeout = min(.01, max(0., self.next_action-time.monotonic()))
                        elif self.running and self.pending is not None:
                            timeout = .001
                        request = self.commands.get(timeout=timeout)
                    except queue.Empty:
                        request = None
                    if request:
                        self.command(request)
                    self.tick()
                except Exception as error:
                    self.record("stop_reason", dict(
                        message=str(error), type=type(error).__name__,
                        trace=getattr(error, "trace", {})))
                    try:
                        self.halt((getattr(self, "cancel_reason", "") or "运行取消")
                                  if isinstance(error, Cancelled) else str(error))
                    except Exception as stop_error:
                        self.emit(f"停止指令失败：{stop_error}；请使用机械臂实体急停", critical=True)
                    normal = isinstance(error, (Cancelled, SegmentComplete))
                    self.emit(("已暂停" if isinstance(error, Cancelled) else f"本段结束：{error}")
                              if normal else f"错误：{error}",
                              homing=False, detail="" if normal else traceback.format_exc())
        finally:
            try:
                self.halt("退出程序")
                self.watchdog.join(timeout=2)
                if self.watchdog.is_alive():
                    raise RuntimeError("独立停止线程尚未退出，保留设备连接直到进程结束")
                self.disconnect()
            except Exception as error:
                self.emit(f"关闭设备失败：{error}", critical=True)
            if self.worker is not None:
                self.worker.terminate()
                self.worker.join(timeout=2)
            self.closed.set()
