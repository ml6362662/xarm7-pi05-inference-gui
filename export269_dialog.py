"""Shared nonblocking data-export dialog for the inference workbenches."""
from pathlib import Path
import json,sys,time
from PySide6 import QtCore,QtGui,QtWidgets as W
ROOT=Path('/home/a/xarm_export269')

class Export269Dialog(W.QDialog):
 def __init__(self,parent=None):
  super().__init__(parent);self.setWindowTitle('训练数据导出 · 269平滑同流程');self.resize(840,620)
  self.process=None;self.buffer='';self.output=None
  box=W.QVBoxLayout(self)
  title=W.QLabel('选择原始数据集 → 导出与平滑 → 自动校验');title.setStyleSheet('font-size:20px;font-weight:bold');box.addWidget(title)
  note=W.QLabel('仅成功且完整的示范 · 30 Hz · 位置改变量≤0.5 mm\n旋转与夹爪不变，保护抓取与持物区间；保留原始数据。完成后生成训练配套文件，不会自动上传或训练。');note.setWordWrap(True);box.addWidget(note)
  self.source=W.QLineEdit();self.destination=W.QLineEdit()
  settings=QtCore.QSettings('xarm','export269')
  default=str(settings.value('source','/home/a/lerobot_datasets/data1109/xarm7_demonstrations'))
  self.source.setText(default)
  for label,edit,handler in [('原始数据集',self.source,self.choose_source),('输出目录',self.destination,self.choose_destination)]:
   row=W.QHBoxLayout();row.addWidget(W.QLabel(label));row.addWidget(edit,1);b=W.QPushButton('选择…');b.clicked.connect(handler);row.addWidget(b);box.addLayout(row)
  self.new_destination()
  self.status=W.QLabel('就绪。选择含 episodes 文件夹的原始数据集。');self.status.setWordWrap(True);box.addWidget(self.status)
  row=W.QHBoxLayout();self.start=W.QPushButton('一键导出、平滑并校验');self.start.clicked.connect(self.run);row.addWidget(self.start)
  self.open_output=W.QPushButton('打开结果目录');self.open_output.setEnabled(False);self.open_output.clicked.connect(lambda:QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(self.output))));row.addWidget(self.open_output)
  self.open_report=W.QPushButton('查看检查报告');self.open_report.setEnabled(False);self.open_report.clicked.connect(self.report);row.addWidget(self.open_report);box.addLayout(row)
  self.log=W.QPlainTextEdit();self.log.setReadOnly(True);self.log.setMaximumBlockCount(1500);box.addWidget(self.log,1)
  self.source.textEdited.connect(lambda _: self.new_destination())
 def new_destination(self):
  name=Path(self.source.text()).parent.name
  self.destination.setText(str(Path('/home/a/lerobot_datasets/exports')/(name+'_269smooth_'+time.strftime('%Y%m%d_%H%M%S'))))
 def choose_source(self):
  p=W.QFileDialog.getExistingDirectory(self,'选择原始数据集',self.source.text())
  if p:self.source.setText(p);self.new_destination()
 def choose_destination(self):
  p=W.QFileDialog.getExistingDirectory(self,'选择输出父目录',str(Path(self.destination.text()).parent))
  if p:self.destination.setText(str(Path(p)/Path(self.destination.text()).name))
 def run(self):
  if self.process and self.process.state()!=QtCore.QProcess.NotRunning:return
  source=Path(self.source.text()).expanduser();dest=Path(self.destination.text()).expanduser()
  if not (source/'episodes').is_dir():self.status.setText('请选择含 episodes 子目录的原始数据集。');return
  if dest.exists() or dest.with_name(dest.name+'.partial').exists():self.status.setText('输出目录已存在，请更换名称，已有结果不会覆盖。');return
  QtCore.QSettings('xarm','export269').setValue('source',str(source))
  self.output=dest;self.log.clear();self.buffer='';self.start.setEnabled(False);self.source.setEnabled(False);self.destination.setEnabled(False);self.open_output.setEnabled(False);self.open_report.setEnabled(False)
  self.status.setText('正在后台导出；随后自动平滑和校验，界面仍可操作。')
  self.process=QtCore.QProcess(self);self.process.setProcessChannelMode(QtCore.QProcess.MergedChannels)
  self.process.readyReadStandardOutput.connect(self.read_output);self.process.finished.connect(self.on_process_finished);self.process.errorOccurred.connect(self.failed)
  self.process.start('/home/a/download/lerobot/.venv/bin/python',['-B','-u',str(ROOT/'export269.py'),str(source),str(dest)])
 def read_output(self):
  data=bytes(self.process.readAllStandardOutput()).decode('utf-8',errors='replace');self.log.moveCursor(QtGui.QTextCursor.End);self.log.insertPlainText(data)
 def failed(self,error):
  if error==QtCore.QProcess.FailedToStart:self.on_process_finished(-1,None)
 def on_process_finished(self,code,status):
  self.read_output();self.start.setEnabled(True);self.source.setEnabled(True);self.destination.setEnabled(True)
  p=self.output/'PREPROCESS_VERIFIED.json';self.open_output.setEnabled(self.output.exists());self.open_report.setEnabled((self.output/'quality_report.json').exists())
  if code==0 and p.exists():
   r=json.loads(p.read_text());q=r['quality_summary'];self.status.setText(f'完成并通过校验：{r["episodes"]} 条 / {r["frames"]:,} 帧；排除 {q["excluded"]} 条，不合格 {q["rejected"]} 条。最大位置改变量 {r["max_shift_mm"]:.4f} mm。')
  else:self.status.setText('处理未通过，未标记为可训练。请查看下方日志；原始数据保留。')
 def report(self):
  q=json.loads((self.output/'quality_report.json').read_text());d=W.QDialog(self);d.setWindowTitle('数据检查报告');d.resize(760,520);b=W.QVBoxLayout(d);t=W.QPlainTextEdit();t.setReadOnly(True)
  t.setPlainText('\n'.join(f'{e["episode_id"]}  {e["decision"]}  {e.get("reason","")}' for e in q['episodes']));b.addWidget(t);d.exec()
 def closeEvent(self,event):
  if self.process and self.process.state()!=QtCore.QProcess.NotRunning:
   self.status.setText('导出正在后台进行，可最小化窗口；完成后再关闭，避免中断处理。');event.ignore()
  else:event.accept()

def open_export269(parent):
 d=getattr(parent,'_export269_dialog',None)
 if d is None:d=Export269Dialog(parent);parent._export269_dialog=d
 d.show();d.raise_();d.activateWindow();return d
