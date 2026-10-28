"""样本是否遵守分析规程。训练前和训练后用同一套。"""

from __future__ import annotations

import json
from dataclasses import dataclass

from ticket_sft.paths import HEADINGS

BANNED_ACOUSTIC = (
    "已完成声学验证",
    "已完成音素级验证",
    "音素层已确认",
    "声学已验证",
)


@dataclass
class Finding:
    level: str
    message: str


def rough_tokens(text: str) -> int:
    """宁可比真实分词估得更大，用来防止结论被 max_length 截断。"""
    special = sum(1 for char in text if ord(char) > 127)
    other = len(text) - special
    return special + (other + 1) // 2


def section(markdown: str, heading: str) -> str:
    marker = f"## {heading}"
    start = markdown.find(marker)
    if start < 0:
        return ""
    start += len(marker)
    rest = markdown[start:]
    next_heading = rest.find("\n## ")
    if next_heading >= 0:
        rest = rest[:next_heading]
    return rest.strip()


def check_conclusion(stdout_obj: dict, conclusion: str) -> list[Finding]:
    findings: list[Finding] = []
    for heading in HEADINGS:
        if not section(conclusion, heading):
            findings.append(Finding("error", f"缺少或为空的标题：## {heading}"))

    dump_present = bool(stdout_obj.get("dump_present"))
    if not dump_present:
        for phrase in BANNED_ACOUSTIC:
            if phrase in conclusion:
                findings.append(Finding("error", f"没有声学 dump，结论里却写了「{phrase}」"))

    if dump_present and _any_false(stdout_obj, "state_map_applied"):
        if "映射" not in section(conclusion, "待核实"):
            findings.append(Finding("error", "state 没有做声学模型映射，待核实里要写明映射"))

    branches = stdout_obj.get("branches")
    if isinstance(branches, list):
        if not branches and "分支" not in section(conclusion, "结论"):
            findings.append(Finding("error", "没有解码分支，结论里要写明没有分支"))
        if branches and any(row.get("target_in_candidates") is False for row in branches):
            verdict = section(conclusion, "结论")
            if not any(phrase in verdict for phrase in ("没有出现", "未出现", "没有进入")):
                findings.append(Finding("error", "正确文本没进候选，结论里要写明没有出现"))
        if any(isinstance(row, dict) and row.get("target_in_candidates") is True for row in branches):
            if "候选" not in section(conclusion, "结论"):
                findings.append(Finding("error", "正确文本在候选里，结论里要写明候选"))

    if stdout_obj.get("score_type") == "槽准" and "槽" not in section(conclusion, "结论"):
        findings.append(Finding("error", "这条按槽准，结论里要写出槽"))

    if _nonempty_lists(stdout_obj, "in_lexicon_but_path_blocked"):
        if "路径" not in section(conclusion, "结论"):
            findings.append(Finding("error", "词在词典里但路径不通，结论里要写明路径"))

    if "回退" not in section(conclusion, "正式门禁"):
        findings.append(Finding("error", "正式门禁要回答有没有回退"))
    return findings


def check_length(text: str, max_length: int) -> list[Finding]:
    estimate = rough_tokens(text)
    if estimate > max_length:
        return [
            Finding(
                "error",
                f"粗估 {estimate} token，超过 max_length {max_length}。缩短问题单，或把 MAX_LENGTH 调大后再训。",
            )
        ]
    if estimate > int(max_length * 0.85):
        return [Finding("warn", f"粗估 {estimate} token，已接近 max_length {max_length}")]
    return []


def _any_false(obj, key: str) -> bool:
    return any(value is False for value in _values(obj, key))


def _nonempty_lists(obj, key: str) -> bool:
    return any(isinstance(value, list) and value for value in _values(obj, key))


def _values(obj, key: str) -> list:
    found = []
    if isinstance(obj, dict):
        if key in obj:
            found.append(obj[key])
        for value in obj.values():
            found.extend(_values(value, key))
    elif isinstance(obj, list):
        for value in obj:
            found.extend(_values(value, key))
    return found


def loads_stdout(stdout: str) -> dict:
    try:
        value = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise ValueError(f"脚本标准输出不是 JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("脚本标准输出的 JSON 顶层必须是对象")
    return value
