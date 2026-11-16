"""Small non-modal annotation browser; all inputs are recorded files."""
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets as W

from run_results import RECORDINGS_ROOT, OUTCOMES, annotate, is_active, list_recordings


class RecordingDialog(W.QDialog):
    saved = QtCore.Signal(str, str)

    def __init__(self, parent=None, root=None):
        super().__init__(parent)
        self.root = Path(root) if root is not None else RECORDINGS_ROOT
        self.selected_path = None
        self.setWindowTitle("推理记录 · 成功与失败标注")
        self.resize(760, 530)
        layout = W.QVBoxLayout(self)
        tip = W.QLabel("请按本次推理的实际结果标注。可以先关闭窗口，稍后再补标；不会自动判断成功或失败。")
        tip.setWordWrap(True)
        layout.addWidget(tip)
        self.records = W.QComboBox()
        self.records.setSizeAdjustPolicy(W.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.records.setMinimumContentsLength(30)
        self.records.currentIndexChanged.connect(self.select_record)
        layout.addWidget(self.records)
        self.details = W.QLabel()
        self.details.setWordWrap(True)
        self.details.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.details)
        row = W.QHBoxLayout()
        row.addWidget(W.QLabel("本次结果"))
        self.outcome = W.QComboBox()
        for key, label in OUTCOMES.items():
            self.outcome.addItem("请选择成功或失败" if key == "pending" else label, key)
        row.addWidget(self.outcome)
        row.addStretch()
        layout.addLayout(row)
        self.note = W.QPlainTextEdit()
        self.note.setPlaceholderText("备注（可不填）：例如未夹住面包、移动时掉落、碰到纸杯、完整任务成功。最多 2000 字。")
        layout.addWidget(self.note)
        self.error = W.QLabel()
        self.error.setWordWrap(True)
        self.error.setStyleSheet("color: #a33423;")
        layout.addWidget(self.error)
        buttons = W.QHBoxLayout()
        self.open_folder = W.QPushButton("打开此条记录文件夹")
        self.open_folder.clicked.connect(self.show_folder)
        self.save_button = W.QPushButton("保存标注")
        self.save_button.clicked.connect(self.save)
        later = W.QPushButton("稍后标注 / 关闭")
        later.clicked.connect(self.close)
        buttons.addWidget(self.open_folder)
        buttons.addStretch()
        buttons.addWidget(later)
        buttons.addWidget(self.save_button)
        layout.addLayout(buttons)

    def refresh(self, selected=None):
        self.records.blockSignals(True)
        self.records.clear()
        self.entries = list_recordings(self.root)
        for entry in self.entries:
            r = entry["result"]
            label = OUTCOMES[r["outcome"]] if r else "读取失败"
            if r and is_active(r):
                label = "记录中"
            elif r and r.get("run_state") == "recording":
                label = "异常退出 · 待标注"
            self.records.addItem(f"{entry['path'].name}  ·  {label}", entry)
        self.records.blockSignals(False)
        index = next((i for i, e in enumerate(self.entries) if selected and e["path"] == Path(selected)), 0)
        self.records.setCurrentIndex(index if self.entries else -1)
        self.select_record()

    def select_record(self, *_):
        entry = self.records.currentData()
        self.selected_path = entry["path"] if entry else None
        r = entry["result"] if entry else None
        editable = bool(r) and not is_active(r)
        self.save_button.setEnabled(editable)
        self.outcome.setEnabled(editable)
        self.note.setEnabled(editable)
        self.open_folder.setEnabled(bool(entry))
        self.error.setText(entry["error"] if entry else "")
        self.note.setPlainText(r.get("note", "") if r else "")
        self.outcome.setCurrentIndex(self.outcome.findData(r["outcome"] if r else "pending"))
        if not entry:
            self.details.setText("暂无记录。开始推理前勾选“记录本次推理”。")
        elif not r:
            self.details.setText(f"结果文件无法读取，原始数据仍保留：\n{self.selected_path}")
        else:
            self.details.setText(
                f"开始：{r.get('started_at', '未知')}\n"
                f"结束：{r.get('ended_at') or '尚无结束时间'}\n"
                f"停止原因：{r.get('stop_reason') or '尚未结束'}"
                f"\n保存位置：{self.selected_path}")
            if r.get("recording_warning"):
                self.error.setText("记录完整性提示：" + r["recording_warning"])

    def show_folder(self):
        if self.selected_path:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(self.selected_path)))

    def save(self):
        if not self.selected_path:
            return
        path = self.selected_path
        try:
            result = annotate(path, self.outcome.currentData(), self.note.toPlainText(), self.root)
        except (OSError, ValueError) as error:
            self.error.setText(str(error))
            return
        self.saved.emit(str(path), result["outcome_label"])
        self.accept()
