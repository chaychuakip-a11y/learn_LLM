"""不接触 GPU 的自测。第 0 课先跑这个。"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from ticket_sft.agent_loop import run_offline
from ticket_sft.basics import lesson_paths
from ticket_sft.audit import audit_jsonl, audit_record
from ticket_sft.build_dataset import build
from ticket_sft.debug import trace_case
from ticket_sft.paths import ROOT
from ticket_sft.plot_log import loss_svg, loss_text
from ticket_sft.preflight import build_swift_args, load_env_file
from ticket_sft.progress import emit, format_bar, format_meter
from ticket_sft.report import build_report
from ticket_sft.sandbox import run_code


def main() -> int:
    stages = [
        ("进度条", _test_progress),
        ("swift 命令", _test_command),
        ("八课", _test_lessons),
        ("loss 文本图", _test_svg),
        ("沙箱拦截", _test_sandbox_blocks),
        ("数据往返", _test_dataset_roundtrip),
    ]
    rough = 0
    total = len(stages)
    for index, (name, fn) in enumerate(stages, start=1):
        emit(f"{format_bar(index - 1, total)}  {name}  开始")
        result = fn()
        if isinstance(result, int):
            rough = result
        emit(f"{format_bar(index, total)}  {name}  通过")
    emit("selftest ok")
    emit(f"最长样本粗估 {rough} token")
    return 0


def _test_progress() -> None:
    bar = format_bar(1, 4)
    if "1/4" not in bar or "#" not in bar or "[----" not in format_bar(0, 4):
        raise SystemExit(f"进度条不对: {bar}")
    meter = format_meter(50, 100)
    if "50%" not in meter:
        raise SystemExit(f"长度条不对: {meter}")


def _test_lessons() -> None:
    paths = lesson_paths()
    if len(paths) != 8:
        raise SystemExit(f"基础课应该是 8 课，实际 {len(paths)}")
    opening = paths[0].read_text(encoding="utf-8")
    if "下一个 token" not in opening:
        raise SystemExit("第 1 课没有讲下一个 token")


def _test_command() -> None:
    config = load_env_file(ROOT / "configs" / "sft_9b.env")
    args = build_swift_args(config, Path("train.jsonl"), Path("val.jsonl"))
    assert args[args.index("--model") + 1] == "Qwen/Qwen3.5-9B"
    assert "ignore_empty_think" in args
    assert "--max_steps" not in args
    assert "--num_train_epochs" in args
    stepped = dict(config)
    stepped["MAX_STEPS"] = "20"
    smoke = build_swift_args(stepped, Path("train.jsonl"), Path("val.jsonl"))
    assert smoke[smoke.index("--max_steps") + 1] == "20"
    assert "--num_train_epochs" not in smoke
    four = load_env_file(ROOT / "configs" / "sft_4b.env")
    assert four["MODEL"] == "Qwen/Qwen3.5-4B"


def _test_svg() -> None:
    svg = loss_svg([(1, 2.2), (2, 1.4)], [(2, 1.8)])
    assert "polyline" in svg
    assert "#1d4e89" in svg
    assert loss_svg([], []) == ""
    text = loss_text([(1, 2.2), (2, 1.4)], [(2, 1.8)])
    if "1.4000" not in text or "#" not in text:
        raise SystemExit(f"loss 文本图不对:\n{text}")
    if loss_text([], []) != "":
        raise SystemExit("空 loss 应该没有文本图")


def _test_sandbox_blocks() -> None:
    case_dir = ROOT / "data" / "example" / "cases" / "zh_contact_004"
    outside = json.dumps(str((ROOT / "pyproject.toml").resolve()))
    blocked = [
        "open('leak.txt', 'w').write('x')\n",
        f"from pathlib import Path\nprint(Path({outside}).read_text(encoding='utf-8')[:20])\n",
        "import socket\nsocket.socket()\n",
    ]
    for code in blocked:
        result = run_code(code, case_dir)
        if result.ok:
            raise SystemExit(f"沙箱没有拦住:\n{code}\n{result.stdout}")
        blob = result.stderr
        if "不允许" not in blob and "PermissionError" not in blob and "沙箱" not in blob:
            raise SystemExit(f"拦截信息看不懂:\n{blob}")
    allowed = run_code("print(open('case.json', encoding='utf-8').read(12))\n", case_dir)
    if not allowed.ok:
        raise SystemExit(f"案例内读取失败:\n{allowed.stderr}")


def _test_dataset_roundtrip() -> int:
    with tempfile.TemporaryDirectory(prefix="ticket-selftest-") as temp:
        out = Path(temp)
        findings = build(
            ROOT / "data" / "example" / "cases",
            ROOT / "data" / "example" / "gold",
            out,
            max_length=4096,
        )
        errors = [item for item in findings if item.level == "error"]
        if errors:
            detail = "\n".join(item.message for item in errors)
            raise SystemExit(f"示例数据没有通过构建:\n{detail}")
        train = [json.loads(line) for line in (out / "train.jsonl").read_text(encoding="utf-8").splitlines()]
        val = [json.loads(line) for line in (out / "val.jsonl").read_text(encoding="utf-8").splitlines()]
        if len(train) != 3 or len(val) != 1:
            raise SystemExit(f"期望 3 条训练、1 条验证，实际 {len(train)} / {len(val)}")
        audits = audit_jsonl(out / "train.jsonl", 4096) + audit_jsonl(out / "val.jsonl", 4096)
        bad = [audit for audit in audits if not audit.ok]
        if bad:
            detail = "\n".join(f"{audit.case_id}: {audit.findings}" for audit in bad)
            raise SystemExit(detail)
        ids = {audit.case_id for audit in audits}
        expected = {"zh_contact_004", "th_poi_001", "de_nav_002", "ar_media_003"}
        if ids != expected:
            raise SystemExit(f"案例编号不对: {ids}")
        _assert_stories(audits)
        zh_line = next(
            line
            for line in (out / "train.jsonl").read_text(encoding="utf-8").splitlines()
            if "zh_contact_004" in line
        )
        broken = json.loads(zh_line)
        broken["messages"][4]["content"] = (
            "## 口径\nx\n## 脚本在查什么\nx\n## 结论\n已完成音素级验证\n## 待核实\n无\n## 正式门禁\n无\n"
        )
        audit = audit_record(broken, 4096)
        if audit.ok or not any("音素" in item.message for item in audit.findings):
            raise SystemExit(f"坏结论没有被拦住: {audit.findings}")
        report = build_report(audits, ROOT / "output")
        if "zh_contact_004" not in report or "要学" not in report:
            raise SystemExit("报告缺内容")
        offline = run_offline(
            ROOT / "data" / "example" / "cases" / "th_poi_001",
            ROOT / "data" / "example" / "gold" / "th_poi_001.py",
        )
        if not offline.ok or "เซ็นทรัล" not in offline.stdout:
            raise SystemExit(offline.stderr)
        zh_trace = trace_case(
            ROOT / "data" / "example" / "cases" / "zh_contact_004",
            ROOT / "data" / "example" / "gold",
            4096,
        )
        zh_audit = next(audit for audit in audits if audit.case_id == "zh_contact_004")
        if not zh_trace.ok or zh_trace.rough != zh_audit.rough:
            raise SystemExit(f"中文调试视图和审计不一致: {zh_trace.rough} vs {zh_audit.rough}\n{zh_trace.text}")
        if "DEL" not in zh_trace.text or "张三" not in zh_trace.text:
            raise SystemExit(zh_trace.text)
        de_trace = trace_case(
            ROOT / "data" / "example" / "cases" / "de_nav_002",
            ROOT / "data" / "example" / "gold",
            4096,
        )
        if "1.5" not in de_trace.text or "映射未做" not in de_trace.text:
            raise SystemExit(de_trace.text)
        return max(audit.rough for audit in audits)


def _assert_stories(audits) -> None:
    by_id = {audit.case_id: json.loads(audit.stdout) for audit in audits}
    zh = by_id["zh_contact_004"]
    if zh["slot_check"][0]["person_in_lexicon"] or zh["slot_check"][0]["person_in_rec"]:
        raise SystemExit("中文联系人样例的槽检查不对")
    if zh["score_type"] != "槽准":
        raise SystemExit("中文样例应该是槽准")
    thai = by_id["th_poi_001"]
    blocked = thai["path_gap"][0]["in_lexicon_but_path_blocked"]
    if blocked != ["เซ็นทรัล"] or thai["branches"][0]["target_in_candidates"]:
        raise SystemExit("泰语样例应该是词在词典里但路径不通，且没进候选")
    german = by_id["de_nav_002"]
    if german["branches"][0]["winner_minus_target"] != 1.5:
        raise SystemExit(f"德语音差不对: {german['branches'][0]}")
    if german["dump"][0]["state_map_applied"] or not german["dump"][0]["matches_winner_text"]:
        raise SystemExit("德语 dump 对照不对")
    arabic = by_id["ar_media_003"]
    if not arabic["branch_table_empty"] or "فيروز" not in arabic["deletions"][0]["tokens"]:
        raise SystemExit("阿拉伯语样例应该是缺词删除，并且没有分支")


if __name__ == "__main__":
    raise SystemExit(main())
