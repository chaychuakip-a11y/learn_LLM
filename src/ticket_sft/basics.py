"""按编号阅读 lessons/ 里的基础课，并在已有样本上补一段实测。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ticket_sft.paths import ROOT, USER_PREAMBLE
from ticket_sft.preflight import load_env_file
from ticket_sft.rules import rough_tokens

LESSONS = ROOT / "lessons"
LEARN_ROLES = {"tool_call", "assistant"}


def lesson_paths() -> list[Path]:
    return sorted(path for path in LESSONS.glob("[0-9][0-9]-*.md") if path.is_file())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="大模型基础课。不需要 GPU。")
    parser.add_argument("number", nargs="?", type=int, help="课号，从 1 开始")
    parser.add_argument("--all", action="store_true", help="按顺序打印全部")
    args = parser.parse_args(argv)
    paths = lesson_paths()
    if not paths:
        print(f"没有找到 {LESSONS}")
        return 1
    if args.all:
        chosen = paths
    elif args.number is None:
        print("大模型基础课，按新手来写。一次只读一课，做完那一课末尾的命令，再打开下一课。\n")
        for index, path in enumerate(paths, start=1):
            title = path.read_text(encoding="utf-8").splitlines()[0].lstrip("# ").strip()
            print(f"  {index}. {title}")
        print("\n查看某一课：python -m ticket_sft.basics 1")
        print("全部打印：  python -m ticket_sft.basics --all")
        return 0
    elif args.number < 1 or args.number > len(paths):
        print(f"课号要在 1 到 {len(paths)} 之间")
        return 1
    else:
        chosen = [paths[args.number - 1]]

    chunks = []
    for path in chosen:
        number = int(path.name[:2])
        text = path.read_text(encoding="utf-8").strip()
        note = _live_note(number)
        if note:
            text += "\n\n" + note
        chunks.append(text)
    payload = ("\n\n----\n\n".join(chunks) + "\n").encode("utf-8")
    sys.stdout.buffer.write(payload)
    return 0


def _live_note(number: int) -> str:
    if number == 2:
        return _token_note()
    if number == 4:
        return _mask_note()
    if number == 6:
        return _step_note()
    return ""


def _records() -> list[dict]:
    records = []
    for name in ("train.jsonl", "val.jsonl"):
        path = ROOT / "data" / "processed" / name
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    return records


def _case_id(record: dict) -> str:
    user = record["messages"][1]["content"]
    if not user.startswith(USER_PREAMBLE):
        return "?"
    case = json.loads(user[len(USER_PREAMBLE) :])
    return str(case.get("case_id", "?"))


def _token_note() -> str:
    records = _records()
    if not records:
        return "你的数据上：还没有 jsonl。先运行 python -m ticket_sft.build_dataset，再看这一课，就能看到每条样本的粗估 token。"
    lines = ["你的数据上，粗估 token（故意偏大，不是分词器真值）："]
    for record in records:
        text = "\n".join(message["content"] for message in record["messages"]) + record.get("tools", "")
        lines.append(f"- {_case_id(record)}：{rough_tokens(text)}")
    return "\n".join(lines)


def _mask_note() -> str:
    records = _records()
    if not records:
        return "你的数据上：先 build_dataset，再看 [要学] 和 [上下文] 各有多少字。"
    record = next((item for item in records if _case_id(item) == "zh_contact_004"), records[0])
    lines = [f"你的数据上，{_case_id(record)} 的字数："]
    for message in record["messages"]:
        role = message["role"]
        kind = "要学" if role in LEARN_ROLES else "上下文"
        lines.append(f"- [{kind}] {role}：{len(message.get('content') or '')} 字")
    return "\n".join(lines)


def _step_note() -> str:
    config = load_env_file(ROOT / "configs" / "sft_9b.env")
    train_path = ROOT / "data" / "processed" / "train.jsonl"
    count = 0
    if train_path.is_file():
        count = sum(1 for line in train_path.read_text(encoding="utf-8").splitlines() if line.strip())
    batch = int(config["PER_DEVICE_BATCH"])
    accum = int(config["GRAD_ACCUM"])
    epochs = int(config["NUM_TRAIN_EPOCHS"])
    micro = max(count, 1)
    steps_per_epoch = max(1, (micro + accum * batch - 1) // (accum * batch)) if count else 1
    if count and count < accum * batch:
        steps_per_epoch = 1
    return (
        "按 configs/sft_9b.env 和当前训练集估算：\n"
        f"- 训练样本 {count or '（还没 build）'} 条，batch {batch}，梯度累积 {accum}，epoch {epochs}\n"
        f"- 样本数少于一次更新所需的 {accum * batch} 条时，每个 epoch 大约更新 {steps_per_epoch} 次\n"
        f"- {epochs} 个 epoch 大约 {steps_per_epoch * epochs} 个 step。冒烟的 20 step 会反复看这几条。"
    )


if __name__ == "__main__":
    raise SystemExit(main())
