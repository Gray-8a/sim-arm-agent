"""结构化事件日志。

一行一个 JSON，字段固定，方便 grep、diff、以及评测脚本回放。
写日志这件事本身是安全设计的一部分：任何被拦截、被审批、被执行的指令
都必须留下可复盘的记录。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class RunLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")
        self._t0 = time.time()

    def event(self, kind: str, **fields: Any) -> dict:
        rec = {
            "ts": round(time.time() - self._t0, 4),
            "kind": kind,
            **fields,
        }
        self._fh.write(json.dumps(rec, ensure_ascii=False, default=_jsonable) + "\n")
        self._fh.flush()
        return rec

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> RunLogger:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _jsonable(obj: Any) -> Any:
    """把 numpy 类型、dataclass 之类降级成可序列化形式。"""
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "value") and hasattr(obj, "name"):  # Enum
        return obj.value
    return str(obj)


def read_jsonl(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out
