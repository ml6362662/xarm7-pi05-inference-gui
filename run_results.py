"""Persistent operator annotations; no robot, camera, or model dependencies."""
from datetime import datetime
import json
import os
from pathlib import Path
import tempfile


RECORDINGS_ROOT = Path(__file__).with_name("推理记录")
OUTCOMES = {"pending": "待标注", "success": "成功", "failure": "失败"}


def timestamp():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def write_json(path, data):
    """Replace only after a complete JSON write; leave the old label on failure."""
    path = Path(path)
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix="." + path.name, suffix=".tmp", delete=False) as stream:
            name = stream.name
            json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if name and os.path.exists(name):
            os.unlink(name)


def read_result(path):
    value = json.loads((Path(path) / "result.json").read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("outcome") not in OUTCOMES:
        raise ValueError("结果文件格式无效")
    return value


def is_active(result):
    if result.get("run_state") != "recording":
        return False
    pid = result.get("owner_pid")
    if type(pid) is not int or pid <= 0:
        return True  # Do not infer that a malformed active record has stopped.
    try:
        os.kill(pid, 0)  # Read-only process existence check.
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    return True


def list_recordings(root=None):
    root = Path(root) if root is not None else RECORDINGS_ROOT
    entries = []
    for path in sorted(root.glob("run_*"), reverse=True):
        if not path.is_dir():
            continue
        try:
            result = read_result(path)
            entries.append(dict(path=path, result=result, error=""))
        except (OSError, ValueError) as error:
            entries.append(dict(path=path, result=None, error=str(error)))
    return entries


def annotate(path, outcome, note="", root=None):
    root = (Path(root) if root is not None else RECORDINGS_ROOT).resolve()
    path = Path(path).resolve()
    if path.parent != root or not path.name.startswith("run_"):
        raise ValueError("只能标注推理记录文件夹中的记录")
    if outcome not in ("success", "failure"):
        raise ValueError("请选择成功或失败")
    if not isinstance(note, str) or len(note) > 2000:
        raise ValueError("备注最多 2000 字")
    result = read_result(path)
    if is_active(result):
        raise ValueError("本次推理尚未结束，请先暂停")
    if result.get("run_state") == "recording":
        result.update(run_state="interrupted", stop_reason="程序异常退出，结束时间未知")
    stamp = timestamp()
    history = list(result.get("annotation_history", []))
    history.append(dict(outcome=outcome, note=note.strip(), annotated_at=stamp))
    result.update(outcome=outcome, outcome_label=OUTCOMES[outcome], note=note.strip(),
                  annotated_at=stamp, annotation_history=history)
    write_json(path / "result.json", result)
    return result
