"""Serializable settings shared by GUI, GPU worker and robot controller."""
from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path

SETTINGS_PATH = Path(__file__).with_name("inference_settings.json")

# label, minimum, maximum, decimals; these limits do not relax hardware guards.
NUMBERS = {
    "home_speed_deg_s": ("返回初始位置速度（°/s）", 5, 30, 1),
    "denoise_steps": ("去噪步数", 1, 20, 0),
    "execute_frames": ("每次执行帧数", 1, 50, 0),
    "seed": ("每次预测使用的种子", 0, 2147483647, 0),
    "control_hz": ("动作频率（Hz）", 5, 30, 0),
    "joint_step_speed": ("关节步进上限（rad/s）", .01, .35, 3),
    "servo_acceleration": ("SDK 加速度（rad/s²）", .05, .75, 2),
    "tcp_step_mm": ("单步 TCP 位移上限（mm）", .1, 5, 2),
    "tracking_mm": ("实测到下发目标距离上限（mm）", 1, 10, 1),
    "inference_drift_deg": ("推理期间关节变化上限（°）", .1, 3, 2),
    "segment_displacement_mm": ("每段最大位移（mm）", 1, 100, 1),
    "segment_path_mm": ("每段最大路径长度（mm）", 1, 150, 1),
    "segment_joint_deg": ("每段最大关节变化（°）", 1, 15, 1),
    "segment_orientation_deg": ("每段最大姿态变化（°）", 1, 20, 1),
    "segment_seconds": ("每段最长时间（秒）", 1, 30, 1),
    "segment_cycles": ("每段最多预测次数", 1, 100, 0),
    "floor_mm": ("TCP 最低高度（mm）", -200, 1000, 1),
    "review_fraction": ("接近模式：预测比例低于此值暂停", .1, 1, 2),
    "close_fraction": ("自动模式：闭合阈值（比例）", 0, .9, 2),
    "open_fraction": ("自动模式：打开阈值（比例）", .1, 1, 2),
    "close_tolerance_mm": ("自动闭合：距原始目标上限（mm）", 1, 10, 1),
    "close_tolerance_deg": ("自动闭合：关节误差上限（°）", .1, 2, 2),
    "gripper_wait_s": ("夹爪开合等待（秒，0为不额外等待）", 0, 10, 1),
    "image_interval_s": ("记录图像最小间隔（秒）", .5, 10, 1),
}


@dataclass(frozen=True)
class InferenceSettings:
    home_speed_deg_s: float = 5.
    denoise_steps: int = 2
    execute_frames: int = 4
    fixed_seed: bool = True
    seed: int = 7
    control_hz: int = 30
    joint_step_speed: float = .1
    servo_acceleration: float = .75
    tcp_step_mm: float = 1.
    tracking_mm: float = 3.
    inference_drift_deg: float = 3.
    diagnostic_constraints: bool = False
    segment_enabled: bool = False
    segment_displacement_mm: float = 50.
    segment_path_mm: float = 60.
    segment_joint_deg: float = 7.
    segment_orientation_deg: float = 10.
    segment_seconds: float = 12.
    segment_cycles: int = 30
    floor_enabled: bool = False
    floor_mm: float = 400.
    gripper_mode: str = "auto"
    review_fraction: float = .8
    close_fraction: float = .4
    open_fraction: float = .6
    close_tolerance_mm: float = 5.
    close_tolerance_deg: float = .5
    gripper_wait_s: float = 0.
    save_images: bool = True
    image_interval_s: float = 1.

    def validate(self):
        for key, (label, lo, hi, decimals) in NUMBERS.items():
            value = getattr(self, key)
            if (type(value) not in (int, float) or not math.isfinite(value)
                    or not lo <= value <= hi or (decimals == 0 and type(value) is not int)):
                raise ValueError(f"{label}必须在 {lo}–{hi} 范围内" + ("，且为整数" if decimals == 0 else ""))
        for key in ("fixed_seed", "diagnostic_constraints", "segment_enabled", "floor_enabled", "save_images"):
            if type(getattr(self, key)) is not bool:
                raise ValueError(f"{key} 必须是布尔值")
        if self.gripper_mode not in MODES:
            raise ValueError("未知夹爪模式")
        if self.close_fraction >= self.open_fraction:
            raise ValueError("闭合阈值必须小于打开阈值")
        if self.segment_path_mm < self.segment_displacement_mm:
            raise ValueError("路径长度上限不能小于位移上限")
        return self

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict) or set(data) - {f.name for f in fields(cls)}:
            raise ValueError("推理设置格式错误或包含未知字段")
        return cls(**data).validate()


MODES = {
    "review": "接近：收拢前暂停，人工确认后闭合",
    "hold": "提起 / 转移：保持夹爪，预测打开时暂停",
    "auto": "自动打开 / 闭合",
}


def preset(name):
    data = InferenceSettings().to_dict()
    # Historical diagnostic scripts may still explicitly request bounded probes.
    data.update(diagnostic_constraints=True, segment_enabled=True,
                gripper_mode="review", inference_drift_deg=.5, gripper_wait_s=4.,
                servo_acceleration=.3)
    if name in ("small", "lift"):
        data.update(segment_displacement_mm=20., segment_path_mm=22.,
                    segment_joint_deg=3., segment_orientation_deg=5.,
                    segment_seconds=4., segment_cycles=6)
    if name == "lift":
        data["gripper_mode"] = "hold"
    elif name == "transfer":
        data["gripper_mode"] = "hold"
    elif name == "auto":
        data["gripper_mode"] = "auto"
    elif name not in ("approach", "small"):
        raise ValueError("未知预设")
    return InferenceSettings.from_dict(data)


def load_settings(path=None):
    path = Path(path) if path is not None else SETTINGS_PATH
    return InferenceSettings.from_dict(json.loads(path.read_text())) if path.exists() else InferenceSettings()


def save_settings(settings, path=None):
    path = Path(path) if path is not None else SETTINGS_PATH
    settings.validate()
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(settings.to_dict(), ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def describe(settings):
    mode = "分段测试" if settings.segment_enabled else "连续推理"
    return (f"{settings.denoise_steps} 步去噪 · {settings.control_hz} Hz · "
            f"每块执行 {settings.execute_frames} 帧 · {mode} · {MODES[settings.gripper_mode]}")


def simple_settings(settings):
    """Explicitly remove probe-only constraints, including persisted old values."""
    from dataclasses import replace
    return replace(settings, diagnostic_constraints=False, segment_enabled=False,
                   floor_enabled=False, gripper_mode="auto", inference_drift_deg=3.,
                   servo_acceleration=.75).validate()
