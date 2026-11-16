#!/usr/bin/env python3
"""Local desktop UI; opening and loading a model do not connect hardware."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import queue
import signal
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "vendor"))
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(ROOT / "vendor/PySide6/Qt/plugins")
os.environ.pop("QT_QPA_FONTDIR", None)

from PySide6 import QtCore, QtGui, QtWidgets as W
from inference_settings import InferenceSettings, load_settings, describe, simple_settings
from settings_dialog import SettingsDialog
from run_results import RECORDINGS_ROOT
from recording_dialog import RecordingDialog
from model_contract import JOINT, TCP, EEFDELTA, POSE_MODES, LABELS
from control_checks import describe_checks

MODEL_ROOT = ROOT.parent / "模型下载_61条_20261027_15000步"
MODEL_ROOTS = (MODEL_ROOT, ROOT.parent / "模型下载_EEFDelta_20261022", ROOT.parent / "模型下载_TCP_20261021", ROOT.parent / "模型下载_20261020",
               ROOT.parent / "模型下载_20261103")


class App(W.QMainWindow):
    def __init__(self):
        super().__init__()
        from engine import Engine
        self.engine = Engine()
        self.closing = self.busy_home = False
        self.busy_gripper = False
        self.starting = False
        self.record_dialog = None
        self.checks_dialog = None
        self.completed_recordings = []
        self.settings_error = ""
        try:
            self.settings = simple_settings(load_settings())
        except (ValueError, OSError, TypeError) as error:
            self.settings = InferenceSettings()
            self.settings_error = f"设置文件无效，请打开推理设置并保存：{error}"
        self.loaded = self.loading = self.connected = self.running = False
        self.selected_loaded = None
        self.model_mode = JOINT
        self.setWindowTitle("π0.5 · XArm7 推理")
        self.resize(1000, 940)
        self.setMinimumSize(850, 840)
        self.setStyleSheet("""
            QMainWindow, QWidget#central { background: #f4f6f8; color: #172b3a; }
            QLabel { color: #172b3a; }
            QPushButton { background: white; border: 1px solid #ccd4dc; border-radius: 6px;
                          padding: 10px 16px; color: #172b3a; }
            QPushButton:hover { background: #e8eef3; }
            QPushButton:disabled { color: #8b99a5; background: #e9edf0; }
            QPushButton#start { background: #087e8b; color: white; border: 0; }
            QPushButton#start:disabled { background: #bdcbd0; color: #edf3f5; }
            QPushButton#pause { background: #fff3dc; border-color: #e6c77f; }
            QComboBox, QPlainTextEdit { background: white; border: 1px solid #ccd4dc;
                                      border-radius: 5px; padding: 6px; color: #172b3a; }
            QGroupBox { border: 1px solid #d7dfe5; border-radius: 6px; margin-top: 12px;
                        padding-top: 10px; font-weight: bold; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; }
        """)
        central = W.QWidget(objectName="central")
        scroll = W.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(W.QFrame.Shape.NoFrame)
        scroll.setWidget(central)
        self.setCentralWidget(scroll)
        layout = W.QVBoxLayout(central)
        layout.setSizeConstraint(W.QLayout.SizeConstraint.SetMinimumSize)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(12)
        title = W.QLabel("π0.5 模型推理")
        title.setStyleSheet("font-size: 25px; font-weight: 600;")
        layout.addWidget(title)
        from export269_dialog import open_export269
        self.export269_button = W.QPushButton('数据导出 · 269平滑同流程')
        self.export269_button.clicked.connect(lambda: open_export269(self))
        layout.addWidget(self.export269_button)
        layout.addWidget(W.QLabel("XArm7 · 双相机 · 复用数据采集程序的设备配置与初始位置"))
        row = W.QHBoxLayout()
        row.addWidget(W.QLabel("模型"))
        self.model = W.QComboBox()
        self.model.setEditable(True)
        self.model.setSizeAdjustPolicy(W.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.model.setMinimumContentsLength(20)
        seen = set()
        for folder in MODEL_ROOTS:
            # 下载打包的目录名形如 mixed_step021000_20261103；同名权重常有硬链接副本，按 inode 去重。
            for path in sorted(folder.glob("pi05*/*.pt")) + sorted(folder.glob("*_step*/*.pt")):
                key = path.stat().st_dev, path.stat().st_ino
                if key not in seen:
                    seen.add(key)
                    self.model.addItem(str(path))
        row.addWidget(self.model, 1)
        self.browse = W.QPushButton("选择文件")
        self.browse.clicked.connect(self.choose)
        row.addWidget(self.browse)
        self.load = W.QPushButton("加载模型")
        self.load.clicked.connect(self.load_model)
        row.addWidget(self.load)
        layout.addLayout(row)
        self.model_status = W.QLabel("尚未加载模型")
        self.model_status.setWordWrap(True)
        layout.addWidget(self.model_status)
        layout.addWidget(W.QLabel("任务描述（已填入训练任务，可修改）"))
        self.prompt = W.QPlainTextEdit((ROOT / "assets/task.txt").read_text().strip())
        self.prompt.setMaximumHeight(80)
        layout.addWidget(self.prompt)
        settings_row = W.QHBoxLayout()
        self.edit_settings = W.QPushButton("推理设置")
        self.edit_settings.clicked.connect(self.configure_inference)
        self.settings_summary = W.QLabel(self.settings_error or describe(self.settings))
        self.settings_summary.setWordWrap(True)
        settings_row.addWidget(self.edit_settings)
        settings_row.addWidget(self.settings_summary, 1)
        self.show_checks = W.QPushButton("生效检查")
        self.show_checks.clicked.connect(self.open_checks)
        settings_row.addWidget(self.show_checks)
        layout.addLayout(settings_row)
        recording_row = W.QHBoxLayout()
        self.record_run = W.QCheckBox("记录本次推理")
        self.record_run.setChecked(True)
        self.record_run.setToolTip("开始前选择。暂停后结束本条记录，可标注成功或失败；再次开始会新建一条。")
        self.review_records = W.QPushButton("标注历史记录")
        self.review_records.clicked.connect(self.open_records)
        self.open_recordings = W.QPushButton("打开记录文件夹")
        self.open_recordings.clicked.connect(self.show_recordings_folder)
        recording_row.addWidget(self.record_run)
        recording_row.addStretch()
        recording_row.addWidget(self.review_records)
        recording_row.addWidget(self.open_recordings)
        layout.addLayout(recording_row)
        self.record_status = W.QLabel("暂停即结束本次记录；成功或失败由你标注，也可以稍后补标。")
        self.record_status.setWordWrap(True)
        layout.addWidget(self.record_status)
        buttons = W.QHBoxLayout()
        self.connect = W.QPushButton("连接设备")
        self.connect.clicked.connect(lambda: self.engine.request("connect"))
        self.start = W.QPushButton("开始推理", objectName="start")
        self.start.clicked.connect(self.start_inference)
        self.pause = W.QPushButton("暂停推理", objectName="pause")
        self.pause.clicked.connect(lambda: self.engine.request("pause"))
        self.home = W.QPushButton("回到初始位置")
        self.home.clicked.connect(self.go_home)
        for button in (self.connect, self.start, self.pause, self.home):
            buttons.addWidget(button)
        layout.addLayout(buttons)
        grip_row = W.QHBoxLayout()
        grip_row.addWidget(W.QLabel("手动夹爪"))
        self.grip_open = W.QPushButton("打开")
        self.grip_close = W.QPushButton("闭合")
        self.grip_open.clicked.connect(lambda: self.manual_gripper(84.))
        self.grip_close.clicked.connect(lambda: self.manual_gripper(0.))
        self.grip_actual = W.QLabel("实测：未连接")
        for widget in (self.grip_open, self.grip_close, self.grip_actual):
            grip_row.addWidget(widget)
        grip_row.addStretch()
        layout.addLayout(grip_row)
        note = W.QLabel("手动夹爪会先暂停推理。Esc：暂停。返回速度可在推理设置中调整；请确认路径净空。")
        note.setWordWrap(True)
        layout.addWidget(note)
        self.status = W.QLabel("就绪：先加载模型，再连接设备。")
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color: #087e8b; font-size: 15px; font-weight: 600;")
        layout.addWidget(self.status)
        info_box = W.QGroupBox("上次有效预测 · 每秒刷新一次")
        info_layout = W.QVBoxLayout(info_box)
        self.inference_summary = W.QLabel("等待模型推理结果")
        self.inference_summary.setWordWrap(True)
        self.inference_summary.setMinimumHeight(36)
        self.inference_details = W.QLabel("预测关节 J1–J7（°）：—\n预测夹爪开度：—")
        self.inference_details.setWordWrap(True)
        self.inference_details.setMinimumHeight(52)
        self.inference_details.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.grip_policy_status = W.QLabel("夹爪自动打开 / 闭合")
        self.grip_policy_status.setWordWrap(True)
        self.grip_policy_status.setMinimumHeight(52)
        info_layout.addWidget(self.inference_summary)
        info_layout.addWidget(self.inference_details)
        info_layout.addWidget(self.grip_policy_status)
        layout.addWidget(info_box)
        previews = W.QHBoxLayout()
        self.previews = {}
        for name, label in (("global", "全局相机"), ("wrist", "腕部相机 · 已旋转 180°")):
            box = W.QGroupBox(label)
            box_layout = W.QVBoxLayout(box)
            view = W.QLabel("连接设备后显示")
            view.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            view.setMinimumSize(300, 150)
            view.setStyleSheet("background: #172b3a; color: #dce6ec; border-radius: 4px; font-weight: normal;")
            box_layout.addWidget(view)
            previews.addWidget(box)
            self.previews[name] = view
        layout.addLayout(previews, 1)
        layout.addWidget(W.QLabel("运行记录"))
        self.log = W.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(80)
        self.log.setMaximumBlockCount(150)
        layout.addWidget(self.log)
        self.log_path = ROOT / "logs" / f"session_{time.strftime('%Y%m%d_%H%M%S')}.log"
        self.log_path.parent.mkdir(exist_ok=True)
        self.model.currentTextChanged.connect(self.buttons)
        self.escape = QtGui.QShortcut(QtGui.QKeySequence("Esc"), self)
        self.escape.activated.connect(lambda: self.engine.request("pause"))
        signal.signal(signal.SIGINT, lambda *_: self.close())
        signal.signal(signal.SIGTERM, lambda *_: self.close())
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.poll)
        self.timer.start(100)
        self.info_timer = QtCore.QTimer(self)
        self.info_timer.timeout.connect(self.refresh_info)
        self.info_timer.start(1000)
        self.preview_timer = QtCore.QTimer(self)
        self.preview_timer.timeout.connect(self.refresh_previews)
        self.preview_timer.start(300)
        self.buttons()
        self.refresh_info()

    def choose(self):
        path, _ = W.QFileDialog.getOpenFileName(self, "选择模型", str(MODEL_ROOT), "PyTorch 权重 (*.pt)")
        if path:
            self.model.setCurrentText(path)

    def load_model(self):
        path = self.model.currentText().strip()
        if not path:
            self.status.setText("请先选择模型文件")
            return
        self.loading, self.loaded = True, False
        self.buttons()
        self.engine.request("load", path=path)

    def start_inference(self):
        if self.settings_error:
            self.status.setText(self.settings_error)
            return
        self.starting = True
        if self.model_mode in POSE_MODES:
            from dataclasses import replace
            self.settings = replace(simple_settings(self.settings), control_hz=30, gripper_wait_s=0.)
            self.settings_summary.setText(describe(self.settings))
        self.buttons()
        self.engine.request("start", prompt=self.prompt.toPlainText().strip(), settings=self.settings.to_dict(),
                            record_run=self.record_run.isChecked())

    def open_records(self, selected=None):
        if self.running or self.starting or self.busy_home or self.busy_gripper or self.closing:
            return
        if isinstance(selected, bool):
            selected = None
        if self.record_dialog is None:
            self.record_dialog = RecordingDialog(self)
            self.record_dialog.saved.connect(self.record_saved)
            self.record_dialog.finished.connect(self.buttons)
        if not self.record_dialog.isVisible():
            self.record_dialog.refresh(selected)
        self.record_dialog.show()  # Non-modal: the GUI heartbeat continues.
        self.record_dialog.raise_()
        self.record_dialog.activateWindow()
        self.buttons()

    def record_saved(self, path, label):
        self.record_status.setText(f"已标注为{label}：{path}")

    def show_recordings_folder(self):
        try:
            RECORDINGS_ROOT.mkdir(parents=True, exist_ok=True)
            if not QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(RECORDINGS_ROOT))):
                self.record_status.setText("请手动打开：" + str(RECORDINGS_ROOT))
        except OSError as error:
            self.record_status.setText("打开记录文件夹失败：" + str(error))

    def configure_inference(self):
        dialog = SettingsDialog(self.settings, self)
        if dialog.exec() == W.QDialog.DialogCode.Accepted:
            self.settings = dialog.settings
            if self.model_mode in POSE_MODES:
                from dataclasses import replace
                self.settings = replace(simple_settings(self.settings), control_hz=30, gripper_wait_s=0.)
            self.settings_error = ""
            self.settings_summary.setText(describe(self.settings))
            self.buttons()
            self.refresh_info()

    def open_checks(self):
        if self.checks_dialog is None:
            self.checks_dialog = W.QDialog(self)
            self.checks_dialog.setWindowTitle("当前生效的控制与检查")
            self.checks_dialog.resize(900, 700)
            layout = W.QVBoxLayout(self.checks_dialog)
            self.checks_text = W.QPlainTextEdit()
            self.checks_text.setReadOnly(True)
            layout.addWidget(self.checks_text)
            close = W.QPushButton("关闭")
            close.clicked.connect(self.checks_dialog.close)
            layout.addWidget(close)
        robot = self.engine.robot
        grip_settings = getattr(robot, "settings", None)
        if grip_settings is None:
            try:
                from hardware import COLLECTION
                grip_settings = json.loads((COLLECTION / "relative_control_settings.json").read_text())
            except (OSError, ValueError):
                grip_settings = {}
        limits = (robot.lower, robot.upper) if hasattr(robot, "lower") and hasattr(robot, "upper") else None
        self.checks_text.setPlainText(describe_checks(self.settings, self.model_mode, grip_settings, limits))
        self.checks_dialog.show()  # Non-modal: heartbeat and pause remain responsive.
        self.checks_dialog.raise_()

    def go_home(self):
        self.busy_home = True
        self.buttons()
        self.engine.request("home", settings=self.settings.to_dict())

    def manual_gripper(self, aperture):
        self.engine.request("gripper", aperture=float(aperture), settings=self.settings.to_dict())

    def buttons(self, *_):
        labeling = self.record_dialog is not None and self.record_dialog.isVisible()
        busy = self.loading or self.running or self.starting or self.busy_home or self.closing or self.busy_gripper
        same = self.selected_loaded == str(Path(self.model.currentText()).expanduser().resolve())
        for widget, enabled in (
            (self.load, not busy), (self.browse, not busy), (self.model, not busy),
            (self.connect, not self.connected and not busy),
            (self.edit_settings, not busy),
            (self.start, self.loaded and same and self.connected and not busy and not labeling and not self.settings_error),
            (self.record_run, not busy),
            (self.review_records, not busy),
            (self.open_recordings, not busy),
            (self.home, self.connected and not self.loading and not self.busy_home and not self.closing and not self.busy_gripper),
            (self.pause, not self.closing),
            (self.grip_open, self.connected and not self.busy_home and not self.loading and not self.closing and not self.busy_gripper),
            (self.grip_close, self.connected and not self.busy_home and not self.loading and not self.closing and not self.busy_gripper),
        ):
            widget.setEnabled(enabled)

    def poll(self):
        self.engine.heartbeat = time.monotonic()
        try:
            while True:
                event = self.engine.events.get_nowait()
                self.loaded, self.loading = event["loaded"], event["loading"]
                self.connected, self.running = event["connected"], event["running"]
                self.starting = False
                self.busy_gripper = event.get("gripper_waiting", False)
                if "homing" in event:
                    self.busy_home = event["homing"]
                if "active_recording" in event:
                    self.record_status.setText("记录中：" + event["active_recording"]
                                              if event["active_recording"] else "本次未记录轨迹和图片")
                if event.get("completed_recording"):
                    path = event["completed_recording"]
                    if path not in self.completed_recordings:
                        self.completed_recordings.append(path)
                    self.record_status.setText("已保存，待标注：" + path)
                    if event.get("recording_warning"):
                        self.record_status.setText(self.record_status.text() + "\n" + event["recording_warning"])
                if "model_path" in event:
                    self.selected_loaded = event["model_path"]
                    previous_mode = self.model_mode
                    self.model_mode = event.get("model_mode", JOINT)
                    if self.model_mode in POSE_MODES:
                        from dataclasses import replace
                        self.settings = replace(simple_settings(self.settings), control_hz=30, gripper_wait_s=0.)
                        self.log.appendPlainText("末端模型：顺序预测，块内30 Hz直接执行前N帧；夹爪开合不中断动作块，不启用RTC。")
                    self.model_status.setText("已加载：" + Path(self.selected_loaded).name
                                              + "\n控制模式：" + LABELS[self.model_mode])
                elif self.loading:
                    self.model_status.setText("正在加载，请稍候…")
                elif not self.loaded:
                    self.model_status.setText("尚未加载模型")
                self.status.setText(event["text"])
                line = f"[{time.strftime('%H:%M:%S')}] {event['text']}"
                self.log.appendPlainText(line)
                with self.log_path.open("a") as f:
                    f.write(line + "\n" + (event.get("detail", "") + "\n" if event.get("detail") else ""))
                self.buttons()
        except queue.Empty:
            pass
        if (self.completed_recordings and not self.running and not self.starting and not self.loading
                and not self.busy_home and not self.busy_gripper and not self.closing
                and (self.record_dialog is None or not self.record_dialog.isVisible())):
            self.open_records(self.completed_recordings.pop(0))
        if self.closing and self.engine.closed.is_set():
            self.timer.stop()
            self.info_timer.stop()
            self.preview_timer.stop()
            W.QApplication.quit()

    def refresh_info(self):
        """Only read CPU snapshots; never issue robot/GPU queries for display."""
        info = self.engine.inference_info
        if info is None:
            self.inference_summary.setText("等待模型推理结果")
            self.inference_details.setText(("预测末端 XYZ（mm）／姿态：—" if self.model_mode in POSE_MODES
                                           else "预测关节 J1–J7（°）：—") + "\n预测夹爪开度：—")
        else:
            age = max(0, time.monotonic() - info["monotonic"])
            stamp = time.strftime("%H:%M:%S", time.localtime(info["received_at"]))
            self.inference_summary.setText(
                f"第 {info['sequence']} 次 · {stamp}（{age:.0f} 秒前） · "
                f"计算 {info['elapsed_ms']:.0f} ms · 预测 {info['shape'][0]} 帧 × {info['shape'][1]} 维，"
                f"每块执行前 {info['execute_frames']} 帧（块内30 Hz，顺序预测）")
            if info.get("model_mode") in POSE_MODES:
                xyz = "   ".join(f"{v:.1f}" for v in info["first_tcp_mm"])
                rpy = "   ".join(f"{v:.1f}" for v in info["first_rpy_deg"])
                self.inference_details.setText(
                    f"首帧预测末端 XYZ（mm，基座坐标）：{xyz}\n姿态 RPY（°，仅用于显示）：{rpy}\n"
                    f"原始预测夹爪：{info['first_gripper_mm']:.1f} mm · "
                    f"位移差：{info['position_delta_mm']:.1f} mm · 姿态差：{info['orientation_delta_deg']:.1f}°")
                if info.get("model_mode") == EEFDELTA:
                    self.inference_details.setText(self.inference_details.text()
                        + "\nEEFDelta：以上为固定观测锚点解码后的绝对目标，增量不逐帧累加。")
            else:
                joints = "   ".join(f"{v:.1f}" for v in info["first_joints_deg"])
                self.inference_details.setText(
                    f"首帧预测关节 J1–J7（°）：{joints}\n"
                    f"首帧原始预测夹爪：{info['first_gripper_mm']:.1f} mm · "
                    f"首帧最大关节变化：{info['max_joint_delta_deg']:.1f}°"
                    "（原始预测，执行仍受限速与限位约束）")
        robot = self.engine.robot
        if robot is None or robot.gripper is None:
            self.grip_actual.setText("实测：未连接")
        else:
            grip = robot.gripper.snapshot()
            age = grip["position_age_s"]
            self.grip_actual.setToolTip(grip["error"] or "")
            if grip["error"]:
                self.grip_actual.setText("夹爪异常（悬停查看）")
            elif age is None or age > .75:
                self.grip_actual.setText("实测：未就绪 / 过期")
            else:
                target = grip.get("committed_target_mm")
                label = {0.: "闭合（0 mm）", 84.: "打开（84 mm）"}.get(target)
                sent = f"已下发：{label} · " if label else ""
                self.grip_actual.setText(f"{sent}实测：{grip['position_mm']:.1f} mm")
        target = getattr(robot, "last_grip", None)
        label = {0.: "闭合（0 mm）", 84.: "打开（84 mm）"}.get(target, "尚无命令，保持现有目标")
        self.grip_policy_status.setText(
            f"推理夹爪目标：{label}\n"
            + ("自动打开 / 闭合 · 与机械臂同步下发" if self.model_mode in POSE_MODES
             else f"自动打开 / 闭合 · 开合后等待 {self.settings.gripper_wait_s:g} 秒"))

    def refresh_previews(self):
        for name, camera in list(self.engine.cameras.items()):
            image, stamp = camera.snapshot(require_fresh=False)
            if image is not None:
                h, w = image.shape[:2]
                qimage = QtGui.QImage(image.data, w, h, image.strides[0], QtGui.QImage.Format.Format_RGB888)
                pixmap = QtGui.QPixmap.fromImage(qimage).scaled(
                    self.previews[name].size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                    QtCore.Qt.TransformationMode.SmoothTransformation)
                self.previews[name].setPixmap(pixmap)

    def closeEvent(self, event):
        if self.engine.closed.is_set():
            event.accept()
            return
        event.ignore()
        if not self.closing:
            self.closing = True
            self.status.setText("正在停止机械臂并关闭设备…")
            self.engine.request("shutdown")
            self.buttons()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ui-smoke", action="store_true")
    parser.add_argument("--load-smoke", action="store_true")
    args = parser.parse_args()
    app = W.QApplication(sys.argv[:1])
    app.setFont(QtGui.QFont("Noto Sans CJK SC", 10))
    window = App()
    window.show()
    if args.ui_smoke:
        QtCore.QTimer.singleShot(750, lambda: window.grab().save(str(ROOT / "界面预览.png")))
        QtCore.QTimer.singleShot(1500, window.close)
    if args.load_smoke:
        started = time.monotonic()
        QtCore.QTimer.singleShot(300, window.load_model)
        timer = QtCore.QTimer(window)
        def check():
            failed = time.monotonic() - started > 2 and not window.loading and "错误" in window.status.text()
            if window.loaded or failed or time.monotonic() - started > 180:
                timer.stop()
                window.grab().save(str(ROOT / "加载后界面.png"))
                (ROOT / "gui_load_check.json").write_text(json.dumps({
                    "loaded": window.loaded, "seconds": time.monotonic() - started,
                    "hardware_connected": window.connected, "message": window.status.text(),
                    "model_mode": window.model_mode,
                }, ensure_ascii=False, indent=2))
                window.close()
        timer.timeout.connect(check)
        timer.start(200)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
