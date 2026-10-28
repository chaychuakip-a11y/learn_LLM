"""离线重放金标脚本，或在训练后把模型接到沙箱上。"""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from pathlib import Path

from ticket_sft.paths import SYSTEM_PROMPT_PATH, USER_PREAMBLE
from ticket_sft.progress import format_bar
from ticket_sft.rules import check_conclusion, loads_stdout
from ticket_sft.sandbox import RunResult, run_code
from ticket_sft.tools_schema import TOOLS


def run_offline(case_dir: Path, script_path: Path, timeout: float = 8) -> RunResult:
    return run_code(script_path.read_text(encoding="utf-8"), case_dir, timeout=timeout)


def run_server(case_dir: Path, base_url: str, model: str, timeout: float = 8) -> int:
    case = json.loads((case_dir / "case.json").read_text(encoding="utf-8"))
    system = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
    messages: list[dict] = [
        {"role": "system", "content": system},
        {"role": "user", "content": USER_PREAMBLE + json.dumps(case, ensure_ascii=False, indent=2)},
    ]
    last_stdout = ""
    for round_index in range(1, 3):
        reply = _chat(base_url, {"model": model, "temperature": 0, "messages": messages, "tools": TOOLS})
        message = reply["choices"][0]["message"]
        messages.append(message)
        tool_calls = message.get("tool_calls") or []
        content = message.get("content") or ""
        if not tool_calls:
            print(content)
            return _judge(last_stdout, content)
        function = tool_calls[0]["function"]
        arguments = function.get("arguments") or "{}"
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                print("工具参数不是 JSON。")
                print(arguments)
                return 1
        code = arguments.get("code") if isinstance(arguments, dict) else None
        if not isinstance(code, str):
            print("工具参数里没有字符串 code，模型没有按约定写脚本。")
            print(message)
            return 1
        print(f"{format_bar(round_index, 2)}  第 {round_index} 轮脚本")
        print(code)
        result = run_code(code, case_dir, timeout=timeout)
        last_stdout = result.stdout
        print("—— 标准输出 ——")
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_calls[0].get("id"),
                "content": json.dumps(
                    {
                        "ok": result.ok,
                        "exit_code": result.exit_code,
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                    },
                    ensure_ascii=False,
                ),
            }
        )
    print("两轮之后模型还在要工具，先停下来看脚本。")
    return 1


def _judge(stdout: str, conclusion: str) -> int:
    if not stdout:
        print("模型没有跑成脚本，没法对照证据。")
        return 1
    try:
        stdout_obj = loads_stdout(stdout)
    except ValueError as exc:
        print(exc)
        return 1
    findings = check_conclusion(stdout_obj, conclusion)
    if not findings:
        print("结论通过了和训练集同一套规程检查。")
        return 0
    for item in findings:
        print(f"[{item.level}] {item.message}")
    return 1 if any(item.level == "error" for item in findings) else 0


def _chat(base_url: str, payload: dict) -> dict:
    url = base_url.rstrip("/") + "/chat/completions"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise SystemExit(f"连不上 {url}。先在内网把 swift deploy 拉起来。{exc}") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="跑通「写脚本 → 沙箱执行 → 结论」")
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--script", type=Path)
    parser.add_argument("--server", default="")
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--timeout", type=float, default=8)
    args = parser.parse_args(argv)
    if args.server:
        return run_server(args.case, args.server, args.model, args.timeout)
    if args.script is None:
        print("离线演练要给 --script。训练完成后改用 --server http://127.0.0.1:8000/v1")
        return 2
    result = run_offline(args.case, args.script, args.timeout)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    conclusion = args.script.with_suffix(".md")
    if conclusion.is_file():
        print("—— 金标结论，这是人写的对照，不是模型生成的 ——")
        print(conclusion.read_text(encoding="utf-8"))
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
