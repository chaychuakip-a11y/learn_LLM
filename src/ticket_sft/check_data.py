"""命令行：检查 jsonl。有错误就返回 1，训练脚本会因此停住。"""

from __future__ import annotations

import argparse
from pathlib import Path

from ticket_sft.audit import audit_jsonl
from ticket_sft.paths import ROOT


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
    for path in paths:
        audits.extend(audit_jsonl(path, args.max_length))
    errors = 0
    warns = 0
    for audit in audits:
        state = "通过" if audit.ok else "失败"
        print(f"[{state}] {audit.case_id}  粗估 {audit.rough} token")
        for item in audit.findings:
            print(f"  [{item.level}] {item.message}")
            if item.level == "error":
                errors += 1
            else:
                warns += 1
    print(f"共 {len(audits)} 条，错误 {errors}，提醒 {warns}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
