"""重放脚本，检查结论有没有超出标准输出。"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from ticket_sft.paths import SYSTEM_PROMPT_PATH, USER_PREAMBLE
from ticket_sft.rules import Finding, check_conclusion, check_length, loads_stdout, rough_tokens
from ticket_sft.sandbox import RunResult, run_code

EXPECTED_ROLES = ("system", "user", "tool_call", "tool_response", "assistant")
LEARN_ROLES = {"tool_call", "assistant"}
SUSPICIOUS = ("subprocess", "socket", "os.system", "open(")


@dataclass
class SampleAudit:
    case_id: str
    findings: list[Finding] = field(default_factory=list)
    code: str = ""
    stdout: str = ""
    stored_stdout: str = ""
    conclusion: str = ""
    learn_chars: dict[str, int] = field(default_factory=dict)
    context_chars: dict[str, int] = field(default_factory=dict)
    rough: int = 0

    @property
    def ok(self) -> bool:
        return not any(item.level == "error" for item in self.findings)


def audit_jsonl(path: Path, max_length: int) -> list[SampleAudit]:
    if not path.is_file():
        return [SampleAudit(path.name, [Finding("error", f"找不到 {path}。先运行 build_dataset")])]
    audits = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            audits.append(SampleAudit(f"{path.name}:{line_no}", [Finding("error", f"JSON 损坏: {exc}")]))
            continue
        audits.append(audit_record(record, max_length, label=f"{path.name}:{line_no}"))
    if not audits:
        audits.append(SampleAudit(path.name, [Finding("error", f"{path} 是空的")]))
    return audits


def audit_record(record: dict, max_length: int, label: str = "sample") -> SampleAudit:
    audit = SampleAudit(label)
    messages = record.get("messages")
    if not isinstance(messages, list) or [item.get("role") for item in messages] != list(EXPECTED_ROLES):
        audit.findings.append(Finding("error", "消息必须正好是 system、user、tool_call、tool_response、assistant"))
        return audit
    extra = set(record) - {"messages", "tools"}
    if extra:
        audit.findings.append(Finding("warn", f"多出来的字段 {sorted(extra)}。swift 只需要 messages 和 tools"))
    if not isinstance(record.get("tools"), str):
        audit.findings.append(Finding("error", "tools 必须是 JSON 字符串"))

    system_now = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
    if messages[0].get("content", "").strip() != system_now:
        audit.findings.append(Finding("warn", "系统提示和 prompts/system.md 不一致。改完提示要重新 build_dataset"))

    user = messages[1].get("content") or ""
    if not user.startswith(USER_PREAMBLE):
        audit.findings.append(Finding("error", "用户消息开头不是框架生成的引导句"))
        return audit
    try:
        case = json.loads(user[len(USER_PREAMBLE) :])
        call = json.loads(messages[2]["content"])
        observed = json.loads(messages[3]["content"])
    except json.JSONDecodeError as exc:
        audit.findings.append(Finding("error", f"案例、工具调用或工具结果不是 JSON: {exc}"))
        return audit

    audit.case_id = str(case.get("case_id") or label)
    code = (call.get("arguments") or {}).get("code")
    if call.get("name") != "run_analysis" or not isinstance(code, str):
        audit.findings.append(Finding("error", "工具调用必须是 run_analysis，并且带 code"))
        return audit
    audit.code = code
    audit.conclusion = messages[4].get("content") or ""
    audit.stored_stdout = observed.get("stdout") or ""
    for role, message in zip(EXPECTED_ROLES, messages):
        size = len(message.get("content") or "")
        if role in LEARN_ROLES:
            audit.learn_chars[role] = size
        else:
            audit.context_chars[role] = size
    joined = "\n".join((message.get("content") or "") for message in messages)
    joined += record.get("tools") or ""
    audit.rough = rough_tokens(joined)
    audit.findings.extend(check_length(joined, max_length))
    for needle in SUSPICIOUS:
        if needle in code:
            audit.findings.append(Finding("warn", f"脚本里出现 {needle}。沙箱会拦截，但最好不要写"))

    replay = _replay(case, code)
    audit.stdout = replay.stdout
    if not replay.ok:
        detail = replay.stderr.strip() or "无标准错误"
        audit.findings.append(Finding("error", f"重放脚本失败\n{detail}"))
        return audit
    if replay.stdout != audit.stored_stdout:
        audit.findings.append(Finding("error", "重放标准输出和样本里保存的不一致。重新运行 build_dataset"))
        return audit
    try:
        stdout_obj = loads_stdout(replay.stdout)
    except ValueError as exc:
        audit.findings.append(Finding("error", str(exc)))
        return audit
    audit.findings.extend(check_conclusion(stdout_obj, audit.conclusion))
    return audit


def _replay(case: dict, code: str) -> RunResult:
    with tempfile.TemporaryDirectory(prefix="ticket-replay-") as temp:
        folder = Path(temp)
        (folder / "case.json").write_text(json.dumps(case, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return run_code(code, folder)
