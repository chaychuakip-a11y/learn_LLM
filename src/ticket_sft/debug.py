"""在终端里看一张问题单：对齐、分支、沙箱耗时、token 占用。"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

from ticket_sft.analysis import base_evidence
from ticket_sft.build_dataset import make_record, sample_text
from ticket_sft.paths import HEADINGS, ROOT, SYSTEM_PROMPT_PATH
from ticket_sft.plot_log import find_history, loss_text, split_series
from ticket_sft.progress import emit, format_bar, format_meter
from ticket_sft.rules import Finding, check_conclusion, check_length, loads_stdout, rough_tokens, section
from ticket_sft.sandbox import run_code


@dataclass
class CaseTrace:
    case_id: str
    ok: bool
    rough: int
    elapsed_ms: float
    text: str


def case_dirs(cases_dir: Path) -> list[Path]:
    return sorted(path.parent for path in cases_dir.glob("*/case.json"))


def trace_case(case_dir: Path, gold_dir: Path, max_length: int, verbose: bool = False) -> CaseTrace:
    case_path = case_dir / "case.json"
    case_id = case_dir.name
    if not case_path.is_file():
        return CaseTrace(case_id, False, 0, 0, f"{case_id} 缺少 case.json\n")
    try:
        case = json.loads(case_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return CaseTrace(case_id, False, 0, 0, f"{case_id} 的 case.json 不是合法 JSON: {exc}\n")

    lines = [
        f"车厂 {case.get('oem')}  语种 {case.get('language')}  {case.get('score_type')}  "
        f"{case.get('runtime')}  资源 {case.get('resource_slot')}",
        f"引擎 {case.get('engine')}  版本 {case.get('version')}  测试集 {case.get('test_set')}",
    ]
    try:
        evidence = base_evidence(case)
    except (KeyError, TypeError, ValueError) as exc:
        lines.append(f"案例结构读不下去: {exc}")
        return CaseTrace(str(case.get("case_id") or case_id), False, 0, 0, "\n".join(lines) + "\n")

    for row in evidence["align"]:
        lines.append(f"对齐 {row['utt_id']}  lab={row['lab']!r}  rec={row['rec']!r}")
        lines.extend(_alignment_lines(row["lab"], row["rec"]))
    for row in evidence["lexicon_miss"]:
        missing = " ".join(row["missing"]) if row["missing"] else "无"
        lines.append(f"词典缺 {row['utt_id']}  {missing}")
    if evidence["branches"]:
        for row in evidence["branches"]:
            lines.extend(_branch_lines(row))
    else:
        lines.append("分支  没有解码分支")
    for utt_id, info in (case.get("wfst") or {}).items():
        missing = " ".join(info.get("missing") or []) or "无"
        state = "路径完整" if info.get("path_complete") else "路径不通"
        lines.append(f"WFST {utt_id}  {state}  缺 {missing}  槽 {info.get('slot')}")
    slots = case.get("slots") or {}
    for utt_id, info in slots.items():
        lines.append(f"槽 {utt_id}  {json.dumps(info, ensure_ascii=False)}")
    if evidence["dump"]:
        for item in evidence["dump"]:
            mapped = "映射已做" if item["state_map_applied"] else "映射未做"
            lines.append(
                f"声学 {item['utt_id']}  best={item['best_path_text']!r}  "
                f"state={item['state_ids']}  {mapped}"
            )
    else:
        lines.append("声学 dump  无")

    script_path = gold_dir / f"{case.get('case_id', case_id)}.py"
    conclusion_path = script_path.with_suffix(".md")
    if not script_path.is_file():
        lines.append("金标脚本  没有")
        return CaseTrace(str(case.get("case_id") or case_id), False, 0, 0, "\n".join(lines) + "\n")

    started = time.perf_counter()
    result = run_code(script_path.read_text(encoding="utf-8"), case_dir)
    elapsed_ms = (time.perf_counter() - started) * 1000
    lines.append(
        f"沙箱  {'通过' if result.ok else '失败'}  exit {result.exit_code}  "
        f"{elapsed_ms:.0f} ms  stdout {len(result.stdout)} 字"
    )
    if not result.ok:
        detail = result.stderr.strip() or "无标准错误"
        lines.append(detail)
        return CaseTrace(str(case.get("case_id") or case_id), False, 0, elapsed_ms, "\n".join(lines) + "\n")

    try:
        stdout_obj = loads_stdout(result.stdout)
    except ValueError as exc:
        lines.append(str(exc))
        return CaseTrace(str(case.get("case_id") or case_id), False, 0, elapsed_ms, "\n".join(lines) + "\n")

    lines.extend(_extra_lines(evidence, stdout_obj))
    findings = []
    conclusion = ""
    if conclusion_path.is_file():
        conclusion = conclusion_path.read_text(encoding="utf-8").strip() + "\n"
        present = [heading for heading in HEADINGS if section(conclusion, heading)]
        lines.append(f"结论标题  {len(present)}/{len(HEADINGS)}  " + " ".join(present))
        missing = [heading for heading in HEADINGS if heading not in present]
        if missing:
            lines.append("缺少标题  " + " ".join(missing))
        findings.extend(check_conclusion(stdout_obj, conclusion))
    else:
        lines.append("金标结论  没有")
        findings.append(Finding("error", "缺少结论"))

    system = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
    code = script_path.read_text(encoding="utf-8")
    record = make_record(case, code, result.stdout, result.stderr, conclusion, system)
    rough = rough_tokens(sample_text(record))
    lines.append(f"长度  {format_meter(rough, max_length)}")
    findings.extend(check_length(sample_text(record), max_length))
    if findings:
        for item in findings:
            lines.append(f"  [{item.level}] {item.message}")
    else:
        lines.append("规程检查  通过")
    if verbose:
        lines.append("—— 脚本标准输出 ——")
        lines.append(result.stdout.rstrip())
    ok = not any(item.level == "error" for item in findings)
    return CaseTrace(str(case.get("case_id") or case_id), ok, rough, elapsed_ms, "\n".join(lines) + "\n")


def _alignment_lines(lab: str, rec: str) -> list[str]:
    lab_tokens = str(lab).split()
    rec_tokens = str(rec).split()
    if not lab_tokens and not rec_tokens:
        return ["    （空）"]
    if lab_tokens == rec_tokens:
        return ["    =   " + " ".join(lab_tokens)]
    lines = []
    matcher = SequenceMatcher(a=lab_tokens, b=rec_tokens)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            lines.extend(f"    =   {token}" for token in lab_tokens[i1:i2])
        elif tag == "delete":
            lines.extend(f"    DEL {token}" for token in lab_tokens[i1:i2])
        elif tag == "insert":
            lines.extend(f"    INS {token}" for token in rec_tokens[j1:j2])
        else:
            left = lab_tokens[i1:i2]
            right = rec_tokens[j1:j2]
            for index in range(max(len(left), len(right))):
                lab_token = left[index] if index < len(left) else None
                rec_token = right[index] if index < len(right) else None
                if lab_token and rec_token:
                    lines.append(f"    SUB {lab_token} -> {rec_token}")
                elif lab_token:
                    lines.append(f"    DEL {lab_token}")
                elif rec_token:
                    lines.append(f"    INS {rec_token}")
    return lines


def _branch_lines(row: dict) -> list[str]:
    matched = [item for item in row["candidates"] if item["equals_lab"]]
    gap = ""
    if len(matched) == 1:
        gap = f"  分差 {round(row['winner_score'] - matched[0]['score'], 1)}"
    entered = "标注在候选里" if row["target_in_candidates"] else "标注未进候选"
    lines = [f"分支 {row['utt_id']}  获胜 {row['winner']}  {entered}{gap}"]
    for candidate in row["candidates"]:
        mark = "*" if candidate["winner"] else " "
        lab = " =标注" if candidate["equals_lab"] else ""
        lines.append(f"  {mark} {candidate['name']:<8} {candidate['score']:<8} {candidate['text']}{lab}")
    return lines


def _extra_lines(evidence: dict, stdout_obj: dict) -> list[str]:
    lines = []
    for row in stdout_obj.get("branches") or []:
        if isinstance(row, dict) and "winner_minus_target" in row:
            lines.append(f"脚本分差 {row.get('utt_id')}  {row['winner_minus_target']}")
    for item in stdout_obj.get("dump") or []:
        if isinstance(item, dict) and "matches_winner_text" in item:
            mark = "和获胜文本一致" if item["matches_winner_text"] else "和获胜文本不一致"
            lines.append(f"脚本声学 {item.get('utt_id')}  {mark}")
    for key in ("slot_check", "path_gap", "deletions"):
        value = stdout_obj.get(key)
        if value:
            lines.append(f"脚本 {key}  {json.dumps(value, ensure_ascii=False)}")
    if "branch_table_empty" in stdout_obj:
        lines.append(f"脚本 branch_table_empty  {stdout_obj['branch_table_empty']}")
    extra = sorted(set(stdout_obj) - set(evidence) - {"slot_check", "path_gap", "deletions", "branch_table_empty"})
    if extra:
        lines.append("脚本还有字段  " + " ".join(extra))
    return lines


def _loss_block(output_dir: Path) -> str:
    history = find_history(output_dir) if output_dir.exists() else []
    text = loss_text(*split_series(history))
    if not text:
        return "训练曲线  还没有 loss。冒烟训练之后，每一行 step 会显示在这里。\n"
    return "训练曲线\n" + text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="在终端查看问题单的对齐、分支、沙箱和长度")
    parser.add_argument("--id", default="", help="只看这一张单，并打印脚本标准输出")
    parser.add_argument("--cases", type=Path, default=ROOT / "data" / "example" / "cases")
    parser.add_argument("--gold", type=Path, default=ROOT / "data" / "example" / "gold")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--output", type=Path, default=ROOT / "output", help="训练日志目录，用来画文本 loss")
    parser.add_argument("--stdout", action="store_true", help="每张单都打印脚本标准输出")
    args = parser.parse_args(argv)

    folders = case_dirs(args.cases)
    if args.id:
        folders = [folder for folder in folders if folder.name == args.id]
        if not folders:
            known = ", ".join(folder.name for folder in case_dirs(args.cases)) or "（空）"
            emit(f"没有 {args.id}。现在有：{known}")
            return 1
    if not folders:
        emit(f"{args.cases} 里没有 case.json")
        return 1

    failed = 0
    longest = 0
    total = len(folders)
    emit(f"问题单调试  {total} 条  max_length {args.max_length}")
    for index, folder in enumerate(folders, start=1):
        emit(f"{format_bar(index - 1, total)}  {folder.name}  正在执行脚本")
        trace = trace_case(folder, args.gold, args.max_length, verbose=bool(args.id) or args.stdout)
        state = "通过" if trace.ok else "失败"
        emit(f"{format_bar(index, total)}  {trace.case_id}  {state}  {trace.elapsed_ms:.0f} ms")
        emit(trace.text.rstrip())
        emit("")
        if not trace.ok:
            failed += 1
        longest = max(longest, trace.rough)
    emit(_loss_block(args.output).rstrip())
    emit(f"汇总  通过 {total - failed}  失败 {failed}  最长粗估 {longest} token")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
