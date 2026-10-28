"""沙箱子进程入口。由 sandbox.py 启动，不要直接当分析脚本用。"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

import ticket_sft.analysis as analysis

sys.modules["analysis"] = analysis


def _allowed(path: Path) -> bool:
    resolved = path.resolve()
    roots = [
        Path(os.environ["CASE_DIR"]).resolve(),
        Path(analysis.__file__).resolve().parent,
        Path(sys.prefix).resolve(),
        Path(getattr(sys, "base_prefix", sys.prefix)).resolve(),
    ]
    for root in roots:
        if resolved == root or root in resolved.parents:
            return True
    return False


def _install_guards() -> None:
    import builtins
    import io
    import socket

    real_open = builtins.open

    def guarded_open(file, mode="r", *args, **kwargs):
        flags = str(mode)
        name = str(file)
        if name.startswith("<") and name.endswith(">"):
            return real_open(file, mode, *args, **kwargs)
        if any(mark in flags for mark in ("w", "a", "x", "+")):
            raise PermissionError("分析脚本只允许读取")
        path = Path(file)
        if not path.is_absolute():
            path = Path.cwd() / path
        if not _allowed(path):
            raise PermissionError(f"不允许读取沙箱外文件: {path}")
        return real_open(file, mode, *args, **kwargs)

    builtins.open = guarded_open
    io.open = guarded_open

    real_os_open = os.open
    write_flags = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT

    def guarded_os_open(path, flags, *args, **kwargs):
        if flags & write_flags:
            raise PermissionError("分析脚本只允许读取")
        if not _allowed(Path(path)):
            raise PermissionError(f"不允许读取沙箱外文件: {path}")
        return real_os_open(path, flags, *args, **kwargs)

    os.open = guarded_os_open

    def blocked(*_args, **_kwargs):
        raise PermissionError("沙箱不允许这个操作")

    os.system = blocked
    os.popen = blocked
    os.remove = blocked
    os.unlink = blocked

    class _BlockedSocket:
        def __init__(self, *_args, **_kwargs):
            raise PermissionError("沙箱无网络")

    socket.socket = _BlockedSocket
    socket.create_connection = blocked

    real_import = builtins.__import__
    banned = {"ctypes", "pty", "subprocess", "socket"}

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        root = name.split(".", 1)[0]
        if root in banned:
            raise PermissionError(f"沙箱不允许导入 {name}")
        return real_import(name, globals, locals, fromlist, level)

    builtins.__import__ = guarded_import


def main() -> int:
    _install_guards()
    code = sys.stdin.read()
    if len(code) > 20_000:
        print("脚本超过 20000 字符", file=sys.stderr)
        return 1
    namespace = {
        "__name__": "__main__",
        "analysis": analysis,
        "json": __import__("json"),
        "CASE_DIR": os.environ["CASE_DIR"],
    }
    try:
        exec(compile(code, "<analysis>", "exec"), namespace, namespace)
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        try:
            traceback.print_exc()
        except Exception as trace_exc:
            print(f"打印堆栈时被拦住: {type(trace_exc).__name__}: {trace_exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
