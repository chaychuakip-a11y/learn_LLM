"""在子进程里执行分析脚本，并挡住写文件、联网和子进程。"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from ticket_sft.paths import SRC


@dataclass
class RunResult:
    ok: bool
    exit_code: int
    stdout: str
    stderr: str


def run_code(code: str, case_dir: Path, timeout: float = 8) -> RunResult:
    case_dir = case_dir.resolve()
    case_file = case_dir / "case.json"
    if not case_file.is_file():
        return RunResult(False, 2, "", f"找不到案例文件: {case_file}")

    env = {
        "PATH": os.environ.get("PATH", ""),
        "CASE_DIR": str(case_dir),
        "PYTHONPATH": str(SRC),
        "PYTHONNOUSERSITE": "1",
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    for key in ("SYSTEMROOT", "LD_LIBRARY_PATH", "LANG", "LC_ALL"):
        if os.environ.get(key):
            env[key] = os.environ[key]

    command = [sys.executable, "-m", "ticket_sft._sandbox_main"]
    popen_kwargs: dict = {}
    if os.name == "posix":
        popen_kwargs["start_new_session"] = True
    try:
        completed = subprocess.run(
            command,
            input=code.encode("utf-8"),
            capture_output=True,
            timeout=timeout,
            env=env,
            cwd=str(case_dir),
            check=False,
            **popen_kwargs,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = _normalize(exc.stdout)
        stderr = _normalize(exc.stderr) + f"\n脚本超时（>{timeout} 秒）"
        return RunResult(False, 124, stdout, stderr.strip())

    stdout = _clip(_normalize(completed.stdout))
    stderr = _clip(_normalize(completed.stderr))
    ok = completed.returncode == 0
    return RunResult(ok, completed.returncode, stdout, stderr)


def _normalize(payload: bytes | None) -> str:
    if not payload:
        return ""
    return payload.decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")


def _clip(text: str, limit: int = 100_000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="对一个案例试跑分析脚本")
    parser.add_argument("--case", required=True, type=Path, help="含 case.json 的目录")
    parser.add_argument("--code", required=True, type=Path, help="Python 脚本路径")
    parser.add_argument("--timeout", type=float, default=8)
    args = parser.parse_args(argv)
    result = run_code(args.code.read_text(encoding="utf-8"), args.case, timeout=args.timeout)
    sys.stdout.write(result.stdout)
    if result.stdout and not result.stdout.endswith("\n"):
        sys.stdout.write("\n")
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    print(f"[sandbox] ok={result.ok} exit={result.exit_code}", file=sys.stderr)
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
