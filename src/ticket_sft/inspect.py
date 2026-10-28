"""在终端展开一条样本：哪一段要学，哪一段只是上下文。"""

from __future__ import annotations

import argparse
import sys

from ticket_sft.audit import SampleAudit, audit_jsonl
from ticket_sft.paths import ROOT
from ticket_sft.progress import format_meter


def load_audits(max_length: int) -> list[SampleAudit]:
    paths = [
        ROOT / "data" / "processed" / "train.jsonl",
        ROOT / "data" / "processed" / "val.jsonl",
    ]
    audits: list[SampleAudit] = []
    for path in paths:
        if path.is_file():
            audits.extend(audit_jsonl(path, max_length))
    return audits


def render(audit: SampleAudit, max_length: int) -> str:
    lines = [
        f"案例 {audit.case_id}  {'通过' if audit.ok else '失败'}  粗估 {audit.rough} token",
        f"长度 {format_meter(audit.rough, max_length)}",
        "",
        "模型要预测的部分标成 [要学]，其余是 [上下文]。loss 只落在 [要学] 上。",
        "标准输出是脚本跑出来的事实，不能让模型把它再学一遍。",
        "",
    ]
    for role, size in audit.context_chars.items():
        lines.append(f"[上下文] {role:<14} {size} 字")
    for role, size in audit.learn_chars.items():
        lines.append(f"[要学]   {role:<14} {size} 字")
    lines.append("")
    if audit.findings:
        lines.append("检查：")
        for item in audit.findings:
            lines.append(f"  [{item.level}] {item.message}")
        lines.append("")
    lines.append("—— 脚本 ——")
    lines.append(audit.code.rstrip())
    lines.append("")
    lines.append("—— 标准输出 ——")
    lines.append(audit.stdout.rstrip())
    lines.append("")
    lines.append("—— 结论 ——")
    lines.append(audit.conclusion.rstrip())
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="查看一条训练样本的学习掩码")
    parser.add_argument("--id", required=True)
    parser.add_argument("--max-length", type=int, default=4096)
    args = parser.parse_args(argv)
    audits = load_audits(args.max_length)
    if not audits:
        print("还没有 data/processed/*.jsonl。先运行 python -m ticket_sft.build_dataset")
        return 1
    matched = [audit for audit in audits if audit.case_id == args.id]
    if not matched:
        known = ", ".join(audit.case_id for audit in audits)
        print(f"没有 {args.id}。现在有：{known}")
        return 1
    sys_text = render(matched[0], args.max_length)
    sys.stdout.buffer.write(sys_text.encode("utf-8", errors="replace"))
    return 0 if matched[0].ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
