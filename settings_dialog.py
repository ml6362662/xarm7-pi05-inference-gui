"""Four common controls; no presets, stage selection or hardware access."""
from dataclasses import replace
from PySide6 import QtWidgets as W
from inference_settings import InferenceSettings, NUMBERS, save_settings, simple_settings


class SettingsDialog(W.QDialog):
    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("推理设置")
        self.resize(510, 320)
        self.settings = simple_settings(settings)
        self.controls = {}
        layout = W.QVBoxLayout(self)
        note = W.QLabel("开始后连续推理，夹爪自动打开或闭合。修改后保存，下次开始生效。")
        note.setWordWrap(True)
        layout.addWidget(note)
        form = W.QFormLayout()
        labels = {
            "denoise_steps": "去噪步数",
            "execute_frames": "每次执行帧数",
            "joint_step_speed": "运行速度（rad/s）",
            "gripper_wait_s": "夹爪等待（秒）",
            "home_speed_deg_s": "返回初始位置速度（°/s）",
        }
        tips = {
            "home_speed_deg_s": "默认5°/s，可选5–30°/s；保存后下次点击返回初始位置时生效。",
            "denoise_steps": "默认 2；改变模型采样的步数。",
            "execute_frames": "默认 4；可选 1–50 帧，50 表示模型本次输出的全部动作。执行后重新观察并推理。",
            "joint_step_speed": "默认 0.1；控制关节运动速度，最大 0.35 rad/s。",
            "gripper_wait_s": "默认 2；开合后等待再继续。0 表示不额外等待。",
        }
        if getattr(parent, "model_mode", None) == "tcp_fixed_anchor_delta":
            labels = {k: v for k, v in labels.items() if k in ("denoise_steps", "execute_frames", "home_speed_deg_s")}
            note.setText("按配置执行前N帧再重新预测；直接下发末端目标，夹爪同步开合。")
        for key, label in labels.items():
            _, lo, hi, decimals = NUMBERS[key]
            widget = W.QDoubleSpinBox() if decimals else W.QSpinBox()
            if decimals:
                widget.setDecimals(2 if key == "joint_step_speed" else 1)
                widget.setSingleStep(.05 if key == "joint_step_speed" else .5)
            widget.setRange(lo, hi)
            widget.setToolTip(tips[key])
            form.addRow(label, widget)
            self.controls[key] = widget
        layout.addLayout(form)
        restore = W.QPushButton("恢复默认")
        restore.clicked.connect(lambda: self.fill(InferenceSettings()))
        layout.addWidget(restore)
        self.error = W.QLabel("")
        self.error.setStyleSheet("color: #b42318;")
        self.error.setWordWrap(True)
        layout.addWidget(self.error)
        buttons = W.QDialogButtonBox(W.QDialogButtonBox.StandardButton.Save | W.QDialogButtonBox.StandardButton.Cancel)
        buttons.button(W.QDialogButtonBox.StandardButton.Save).setText("保存")
        buttons.button(W.QDialogButtonBox.StandardButton.Cancel).setText("取消")
        buttons.accepted.connect(self.save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.fill(settings)

    def fill(self, settings):
        self.settings = simple_settings(settings)
        for key, widget in self.controls.items():
            widget.setValue(getattr(self.settings, key))

    def values(self):
        return simple_settings(replace(self.settings, **{
            key: widget.value() for key, widget in self.controls.items()}))

    def save(self):
        try:
            self.settings = self.values()
            save_settings(self.settings)
        except (ValueError, OSError) as error:
            self.error.setText(str(error))
            return
        self.accept()
