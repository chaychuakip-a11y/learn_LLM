"""命令行：检查 jsonl。有错误就返回 1，训练脚本会因此停住。"""

from __future__ import annotations

import argparse
from pathlib import Path

from ticket_sft.audit import audit_jsonl
from ticket_sft.paths import ROOT
from ticket_sft.progress import emit, format_bar


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="检查训练集是否能重放、结论是否越过证据")
    parser.add_argument("jsonl", nargs="*", type=Path)
    parser.add_argument("--max-length", type=int, default=4096)
    args = parser.parse_args(argv)
    paths = args.jsonl or [
        ROOT / "data" / "processed" / "train.jsonl",
        ROOT / "data" / "processed" / "val.jsonl",
    ]
    audits = []

    def _progress(index: int, total: int, audit, seconds: float) -> None:
        state = "通过" if audit.ok else "失败"
        emit(
            f"{format_bar(index, total)}  {audit.case_id}  {state}  "
            f"{seconds * 1000:.0f} ms  粗估 {audit.rough} token"
        )

    for path in paths:
        emit(f"检查 {path}")
        audits.extend(audit_jsonl(path, args.max_length, on_progress=_progress))
    errors = 0
    warns = 0
    for audit in audits:
        for item in audit.findings:
            emit(f"  [{item.level}] {audit.case_id}  {item.message}")
            if item.level == "error":
                errors += 1
            else:
                warns += 1
    emit(f"{format_bar(len(audits), len(audits) or 1)}  共 {len(audits)} 条，错误 {errors}，提醒 {warns}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
