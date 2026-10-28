"""终端进度条。只用 ASCII，重定向到日志时也一行一条。"""

from __future__ import annotations

import sys


def format_bar(done: int, total: int, width: int = 20) -> str:
    if total <= 0:
        return "[" + "-" * width + "] 0/0"
    done = max(0, min(done, total))
    filled = round(width * done / total)
    return "[" + "#" * filled + "-" * (width - filled) + f"] {done}/{total}"


def format_meter(value: int, limit: int, width: int = 20) -> str:
    if limit <= 0:
        return f"[{'-' * width}] {value}"
    ratio = min(1.0, value / limit)
    filled = round(width * ratio)
    return "[" + "#" * filled + "-" * (width - filled) + f"] {value}/{limit} ({ratio:.0%})"


def emit(text: str) -> None:
    if not text.endswith("\n"):
        text += "\n"
    stream = sys.stdout
    stream.flush()
    data = text.encode("utf-8", errors="replace")
    buffer = getattr(stream, "buffer", None)
    if buffer is None:
        stream.write(text)
        stream.flush()
        return
    buffer.write(data)
    buffer.flush()
