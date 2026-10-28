"""把案例、金标脚本和金标结论收成 ms-swift 能读的 jsonl。"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ticket_sft.paths import HEADINGS, ROOT, SYSTEM_PROMPT_PATH, USER_PREAMBLE
from ticket_sft.progress import emit, format_bar, format_meter
from ticket_sft.rules import Finding, check_conclusion, check_length, loads_stdout, rough_tokens
from ticket_sft.sandbox import run_code
from ticket_sft.tools_schema import tools_string


REQUIRED_CASE = ("case_id", "oem", "language", "runtime", "score_type", "lexicon", "utterances")
REQUIRED_UTT = ("utt_id", "lab", "rec")


def make_record(case: dict, code: str, stdout: str, stderr: str, conclusion: str, system: str) -> dict:
    return {
        "tools": tools_string(),
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": USER_PREAMBLE + json.dumps(case, ensure_ascii=False, indent=2)},
            {
                "role": "tool_call",
                "content": json.dumps(
                    {"name": "run_analysis", "arguments": {"code": code}},
                    ensure_ascii=False,
                ),
            },
            {
                "role": "tool_response",
                "content": json.dumps(
                    {
                        "ok": True,
                        "exit_code": 0,
                        "stdout": stdout,
                        "stderr": stderr,
                    },
                    ensure_ascii=False,
                ),
            },
            {"role": "assistant", "content": conclusion},
        ],
    }


def sample_text(record: dict) -> str:
    return "\n".join(message["content"] for message in record["messages"]) + record["tools"]


def build(
    cases_dir: Path,
    gold_dir: Path,
    out_dir: Path,
    max_length: int = 4096,
    on_progress=None,
) -> list[Finding]:
    system = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
    scripts = sorted(gold_dir.glob("*.py"))
    if not scripts:
        return [Finding("error", f"{gold_dir} 里没有 .py 脚本")]

    grouped: dict[str, list[dict]] = {"train": [], "val": []}
    findings: list[Finding] = []
    total = len(scripts)
    for index, script_path in enumerate(scripts, start=1):
        case_id = script_path.stem
        if on_progress:
            on_progress({"index": index, "total": total, "case_id": case_id, "phase": "start"})
        started = time.perf_counter()
        record, sample_findings = _one(case_id, cases_dir, gold_dir, system, max_length)
        elapsed = time.perf_counter() - started
        findings.extend(sample_findings)
        rough = rough_tokens(sample_text(record)) if record is not None else 0
        if on_progress:
            status = "失败" if any(item.level == "error" for item in sample_findings) else "通过"
            on_progress(
                {
                    "index": index,
                    "total": total,
                    "case_id": case_id,
                    "phase": "done",
                    "status": status,
                    "seconds": elapsed,
                    "rough": rough,
                }
            )
        if record is not None:
            split = _split(gold_dir / f"{case_id}.meta.json")
            grouped[split].append(record)

    errors = [item for item in findings if item.level == "error"]
    if errors:
        return findings
    if not grouped["train"] or not grouped["val"]:
        return findings + [Finding("error", "训练集和验证集都至少要有 1 条。在 meta.json 里写 split。")]

    out_dir.mkdir(parents=True, exist_ok=True)
    for split, records in grouped.items():
        path = out_dir / f"{split}.jsonl"
        lines = [json.dumps(record, ensure_ascii=False) for record in records]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return findings


def _split(meta_path: Path) -> str:
    if not meta_path.is_file():
        return "train"
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    split = payload.get("split", "train")
    if split not in ("train", "val"):
        raise ValueError(f"{meta_path} 的 split 只能是 train 或 val")
    return split


def _one(
    case_id: str,
    cases_dir: Path,
    gold_dir: Path,
    system: str,
    max_length: int,
) -> tuple[dict | None, list[Finding]]:
    findings: list[Finding] = []
    case_path = cases_dir / case_id / "case.json"
    script_path = gold_dir / f"{case_id}.py"
    conclusion_path = gold_dir / f"{case_id}.md"
    if not case_path.is_file():
        return None, [Finding("error", f"缺少案例 {case_path}")]
    if not conclusion_path.is_file():
        return None, [Finding("error", f"缺少结论 {conclusion_path}")]

    case = json.loads(case_path.read_text(encoding="utf-8"))
    for key in REQUIRED_CASE:
        if key not in case:
            findings.append(Finding("error", f"{case_id} 缺少字段 {key}"))
    if case.get("case_id") not in (None, case_id):
        findings.append(Finding("error", f"{case_id} 目录名和 case_id 不一致"))
    for index, utterance in enumerate(case.get("utterances") or []):
        for key in REQUIRED_UTT:
            if key not in utterance:
                findings.append(Finding("error", f"{case_id} 第 {index} 句缺少 {key}"))
    conclusion = conclusion_path.read_text(encoding="utf-8").strip() + "\n"
    for heading in HEADINGS:
        if f"## {heading}" not in conclusion:
            findings.append(Finding("error", f"{case_id} 结论缺少标题 ## {heading}"))
    if any(item.level == "error" for item in findings):
        return None, findings

    result = run_code(script_path.read_text(encoding="utf-8"), case_path.parent)
    if not result.ok:
        detail = result.stderr.strip() or "脚本没有标准错误输出"
        return None, findings + [Finding("error", f"{case_id} 脚本失败\n{detail}")]
    try:
        stdout_obj = loads_stdout(result.stdout)
    except ValueError as exc:
        return None, findings + [Finding("error", f"{case_id} {exc}")]
    for item in check_conclusion(stdout_obj, conclusion):
        findings.append(Finding(item.level, f"{case_id} {item.message}"))
    if any(item.level == "error" for item in findings):
        return None, findings

    code = script_path.read_text(encoding="utf-8")
    record = make_record(case, code, result.stdout, result.stderr, conclusion, system)
    text = sample_text(record)
    for item in check_length(text, max_length):
        findings.append(Finding(item.level, f"{case_id} {item.message}"))
    if any(item.level == "error" for item in findings):
        return None, findings
    return record, findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="由案例和金标生成 train.jsonl / val.jsonl")
    parser.add_argument("--cases", type=Path, default=ROOT / "data" / "example" / "cases")
    parser.add_argument("--gold", type=Path, default=ROOT / "data" / "example" / "gold")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--max-length", type=int, default=4096)
    args = parser.parse_args(argv)

    def _progress(event: dict) -> None:
        if event["phase"] == "start":
            emit(f"{format_bar(event['index'] - 1, event['total'])}  {event['case_id']}  正在执行脚本")
            return
        meter = format_meter(event["rough"], args.max_length) if event["rough"] else "没有粗估"
        emit(
            f"{format_bar(event['index'], event['total'])}  {event['case_id']}  {event['status']}  "
            f"{event['seconds'] * 1000:.0f} ms  {meter}"
        )

    findings = build(args.cases, args.gold, args.out, args.max_length, on_progress=_progress)
    for item in findings:
        emit(f"[{item.level}] {item.message}")
    errors = [item for item in findings if item.level == "error"]
    if errors:
        print(f"没有写出训练集。先改掉 {len(errors)} 个错误。")
        return 1
    print(f"已写入 {args.out / 'train.jsonl'} 和 {args.out / 'val.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
