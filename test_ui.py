"""UI checks use only cached data and never create hardware connections."""
import os
import queue
import threading
import time
from types import SimpleNamespace

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import pytest
import app as ui
import engine


@pytest.fixture
def window(monkeypatch):
    qt = ui.W.QApplication.instance() or ui.W.QApplication([])
    closed = threading.Event()
    closed.set()
    requests = []
    stub = SimpleNamespace(
        events=queue.Queue(), inference_info=None, robot=None, cameras={},
        closed=closed, heartbeat=0,
        request=lambda kind, **values: requests.append((kind, values)),
    )
    monkeypatch.setattr(engine, "Engine", lambda: stub)
    w = ui.App()
    w._test_requests = requests
    yield w
    w.timer.stop()
    w.info_timer.stop()
    w.preview_timer.stop()
    w.close()


def test_manual_controls_only_send_explicit_physical_targets(window):
    assert not window.grip_open.isEnabled()
    window.connected = True
    window.running = True
    window.buttons()
    window.grip_open.click()
    window.grip_close.click()
    assert not hasattr(window, "grip_target")
    assert not hasattr(window, "grip_apply")
    assert window._test_requests == [
        ("gripper", {"aperture": 84., "settings": window.settings.to_dict()}),
        ("gripper", {"aperture": 0., "settings": window.settings.to_dict()}),
    ]
    window.busy_home = True
    window.buttons()
    assert not window.grip_open.isEnabled()



def test_checks_inventory_is_nonmodal_and_does_not_query_hardware(window):
    from model_contract import TCP, JOINT
    window.engine.robot = SimpleNamespace(
        state=lambda: pytest.fail("inventory queried hardware"),
        settings={"gripper_speed_mm_s": 50, "gripper_force": 2},
        lower=[-1.] * 7, upper=[1.] * 7,
    )
    window.model_mode = TCP
    window.open_checks()
    assert not window.checks_dialog.isModal()
    assert window.checks_text.isReadOnly()
    text = window.checks_text.toPlainText()
    assert "已删除：预测关节目标" in text
    assert "0.12 rad" in text and "TCP 专用" in text
    assert "分段限制：关闭" in text and "最低高度：关闭" in text
    assert "J1 [-57.3, 57.3]" in text
    assert f"每次执行 {window.settings.execute_frames} 帧" in text
    window.model_mode = JOINT
    window.open_checks()
    assert "TCP 专用" not in window.checks_text.toPlainText()
    assert not window._test_requests


def test_prediction_display_uses_cached_data_without_commands(window):
    window.engine.inference_info = dict(
        sequence=3, received_at=time.time(), monotonic=time.monotonic(),
        elapsed_ms=111., shape=(50, 8), execute_frames=14,
        first_joints_deg=[1, 2, 3, 4, 5, 6, 7],
        first_gripper_mm=42., max_joint_delta_deg=5.,
    )
    window.engine.robot = SimpleNamespace(
        state=lambda: pytest.fail("display issued a hardware query"),
        gripper=SimpleNamespace(snapshot=lambda: dict(
            error="", position_age_s=.1, position_mm=41.5)),
    )
    window.refresh_info()
    assert "111 ms" in window.inference_summary.text()
    assert "42.0 mm" in window.inference_details.text()
    assert "41.5 mm" in window.grip_actual.text()
    assert not window._test_requests
    assert window.info_timer.interval() >= 1000
    assert window.timer.interval() <= 100


def test_tcp_prediction_display_and_model_discovery(window):
    from model_contract import TCP
    paths = [window.model.itemText(i) for i in range(window.model.count())]
    assert any("模型下载_TCP_20261021" in p for p in paths)
    assert any("模型下载_20261020" in p for p in paths)
    window.model_mode = TCP
    window.refresh_info()
    assert "末端" in window.inference_details.text()
    window.engine.inference_info = dict(
        model_mode=TCP, sequence=1, received_at=time.time(), monotonic=time.monotonic(),
        elapsed_ms=90., shape=(50, 8), execute_frames=4, first_tcp_mm=[200, 300, 400],
        first_rpy_deg=[180, 0, 0], first_gripper_mm=84., position_delta_mm=10., orientation_delta_deg=2.)
    window.refresh_info()
    assert "200.0" in window.inference_details.text()
    assert "RPY" in window.inference_details.text()
    assert "J1" not in window.inference_details.text()
    assert not window._test_requests


def test_eefdelta_discovery_defaults_and_cached_display(window):
    from model_contract import EEFDELTA
    from dataclasses import replace
    paths = [window.model.itemText(i) for i in range(window.model.count())]
    delta = [p for p in paths if "模型下载_EEFDelta_20261022" in p]
    assert len(delta) == 3
    window.settings = replace(window.settings, execute_frames=16)
    window.engine.events.put(dict(text="loaded", model_path=delta[0], model_mode=EEFDELTA,
        loaded=True, loading=False, connected=False, running=False))
    window.poll()
    assert window.settings.execute_frames == 5
    window.settings = replace(window.settings, execute_frames=12)
    window.engine.events.put(dict(text="loaded", model_path=delta[1], model_mode=EEFDELTA,
        loaded=True, loading=False, connected=False, running=False))
    window.poll()
    assert window.settings.execute_frames == 12  # Keep explicit changes within this mode.
    window.engine.inference_info = dict(model_mode=EEFDELTA, sequence=1,
        received_at=time.time(), monotonic=time.monotonic(), elapsed_ms=100., shape=(50,7),
        execute_frames=5, first_tcp_mm=[200,300,400],first_rpy_deg=[180,0,0],
        first_gripper_mm=84.,position_delta_mm=10.,orientation_delta_deg=2.)
    window.refresh_info()
    assert "50 帧 × 7 维" in window.inference_summary.text()
    assert "固定观测锚点" in window.inference_details.text()
    window.open_checks()
    assert "输入10维" in window.checks_text.toPlainText()
    assert not window._test_requests


def test_old_prediction_retained_with_age_and_cleared_when_model_resets(window):
    window.engine.inference_info = dict(
        sequence=1, received_at=time.time()-20, monotonic=time.monotonic()-20,
        elapsed_ms=100., shape=(50, 8), execute_frames=14,
        first_joints_deg=[0]*7, first_gripper_mm=0., max_joint_delta_deg=0.,
    )
    window.refresh_info()
    assert "20 秒前" in window.inference_summary.text()
    window.engine.inference_info = None
    window.refresh_info()
    assert "等待" in window.inference_summary.text()
    assert "—" in window.inference_details.text()


def test_binary_target_and_acknowledged_command_are_distinct_from_measurement(window):
    window.engine.robot = SimpleNamespace(
        last_grip=0.,
        gripper=SimpleNamespace(snapshot=lambda: dict(
            error="", position_age_s=.1, position_mm=5., committed_target_mm=0.)),
    )
    window.refresh_info()
    assert "推理夹爪目标：闭合（0 mm）" in window.grip_policy_status.text()
    assert "已下发：闭合（0 mm）" in window.grip_actual.text()
    assert "实测：5.0 mm" in window.grip_actual.text()
    assert not window._test_requests


def test_pending_target_not_falsely_displayed_as_acknowledged(window):
    window.engine.robot = SimpleNamespace(
        last_grip=0.,
        gripper=SimpleNamespace(snapshot=lambda: dict(
            error="", position_age_s=.1, position_mm=84., committed_target_mm=84.)),
    )
    window.refresh_info()
    assert "推理夹爪目标：闭合" in window.grip_policy_status.text()
    assert "已下发：打开" in window.grip_actual.text()
    window.engine.robot.last_grip = None
    window.refresh_info()
    assert "保持现有目标" in window.grip_policy_status.text()


def test_simple_settings_save_and_start_payload(window, monkeypatch, tmp_path):
    from settings_dialog import SettingsDialog
    import settings_dialog
    from inference_settings import load_settings, save_settings
    path = tmp_path / "settings.json"
    monkeypatch.setattr(settings_dialog, "save_settings", lambda value: save_settings(value, path))
    dialog = SettingsDialog(window.settings, window)
    dialog.controls["denoise_steps"].setValue(3)
    dialog.controls["execute_frames"].setValue(50)
    dialog.controls["joint_step_speed"].setValue(.2)
    dialog.controls["gripper_wait_s"].setValue(1.5)
    dialog.save()
    assert dialog.result() == ui.W.QDialog.DialogCode.Accepted
    window.settings = dialog.settings
    assert load_settings(path) == window.settings
    window.start_inference()
    kind, data = window._test_requests[-1]
    assert kind == "start" and data["settings"]["denoise_steps"] == 3
    assert data["settings"]["execute_frames"] == 50
    assert data["settings"]["joint_step_speed"] == .2
    assert data["settings"]["gripper_wait_s"] == 1.5
    assert not data["settings"]["segment_enabled"]
    dialog.close()


def test_simple_dialog_has_only_four_fields_and_save_errors_are_visible(window, monkeypatch):
    from settings_dialog import SettingsDialog
    import settings_dialog
    def fail(value):
        raise OSError("无法保存")
    monkeypatch.setattr(settings_dialog, "save_settings", fail)
    dialog = SettingsDialog(window.settings, window)
    assert set(dialog.controls) == {"denoise_steps", "execute_frames", "joint_step_speed", "gripper_wait_s"}
    assert not dialog.findChildren(ui.W.QTabWidget)
    assert not dialog.findChildren(ui.W.QComboBox)
    dialog.save()
    assert "无法保存" in dialog.error.text()
    assert dialog.result() != ui.W.QDialog.DialogCode.Accepted
    dialog.close()


def test_settings_locked_during_motion_and_gripper_wait(window):
    window.running = True
    window.buttons()
    assert not window.edit_settings.isEnabled()
    window.running = False
    window.busy_gripper = True
    window.connected = True
    window.buttons()
    assert not window.edit_settings.isEnabled()
    assert not window.start.isEnabled()
    assert not window.grip_open.isEnabled()
    assert window.pause.isEnabled()


def test_bad_saved_settings_disable_start_until_corrected(window):
    window.settings_error = "设置无效"
    window.start_inference()
    assert not window._test_requests
    assert "设置无效" in window.status.text()
