"""训练前的环境检查，并把真正要执行的 swift 命令写到文件里。"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
from pathlib import Path

from ticket_sft.paths import ROOT

EXPLAINS = (
    "底模权重冻结，只训练 LoRA。Qwen3.5-9B 的 bf16 LoRA 大约要 22GB 显存。",
    "模型要学的是写脚本和写结论。标准输出来自沙箱，不进入 loss。",
    "add_non_thinking_prefix 让回答带上空的思考块，ignore_empty_think 让这个空块不计 loss。",
    "freeze_vit 冻住视觉塔。这批数据是文本问题单，不训练看图。",
    "max_length 设得比样本短时，结论会被截掉，模型学到的是半句话。check_data 会拦住这件事。",
    "示例验证集只有 1 条。eval loss 下降只说明没训崩。",
    "学习率 1e-4、rank 8 来自 Qwen3.5 的官方 LoRA 示例。loss 振荡就把它降到 5e-5。",
    "group_by_length 关掉了。样本很少时打开它，loss 曲线会跳，不容易看。",
)


def load_env_file(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        key, sep, value = line.partition("=")
        if not sep:
            raise ValueError(f"配置行无法解析: {raw}")
        data[key.strip()] = value.strip().strip('"').strip("'")
    return data


def merged_config(path: Path) -> dict[str, str]:
    config = load_env_file(path)
    for key, value in config.items():
        override = os.environ.get(key)
        if override:
            config[key] = override
    return config


def build_swift_args(config: dict[str, str], train_path: Path, val_path: Path) -> list[str]:
    args = [
        "swift",
        "sft",
        "--model",
        config["MODEL"],
        "--tuner_type",
        "lora",
        "--dataset",
        str(train_path),
        "--val_dataset",
        str(val_path),
        "--torch_dtype",
        config["TORCH_DTYPE"],
        "--lora_rank",
        config["LORA_RANK"],
        "--lora_alpha",
        config["LORA_ALPHA"],
        "--target_modules",
        config["TARGET_MODULES"],
        "--learning_rate",
        config["LEARNING_RATE"],
        "--per_device_train_batch_size",
        config["PER_DEVICE_BATCH"],
        "--per_device_eval_batch_size",
        "1",
        "--gradient_accumulation_steps",
        config["GRAD_ACCUM"],
        "--max_length",
        config["MAX_LENGTH"],
        "--warmup_ratio",
        config["WARMUP_RATIO"],
        "--logging_steps",
        config["LOGGING_STEPS"],
        "--save_steps",
        config["SAVE_STEPS"],
        "--eval_steps",
        config["EVAL_STEPS"],
        "--eval_strategy",
        "steps",
        "--save_total_limit",
        config["SAVE_TOTAL_LIMIT"],
        "--seed",
        config["SEED"],
        "--output_dir",
        config["OUTPUT_DIR"],
        "--add_non_thinking_prefix",
        config["ADD_NON_THINKING_PREFIX"],
        "--loss_scale",
        config["LOSS_SCALE"],
        "--gradient_checkpointing",
        config["GRADIENT_CHECKPOINTING"],
        "--split_dataset_ratio",
        "0",
        "--dataset_num_proc",
        config.get("DATASET_NUM_PROC") or "1",
        "--dataloader_num_workers",
        config.get("DATALOADER_NUM_WORKERS") or "0",
        "--group_by_length",
        "false",
        "--report_to",
        "tensorboard",
        "--model_author",
        config["MODEL_AUTHOR"],
        "--model_name",
        config["MODEL_NAME"],
    ]
    if config.get("MAX_STEPS"):
        args.extend(["--max_steps", config["MAX_STEPS"]])
    else:
        args.extend(["--num_train_epochs", config["NUM_TRAIN_EPOCHS"]])
    if config.get("FREEZE_VIT") == "true":
        args.extend(["--freeze_vit", "true", "--freeze_aligner", "true"])
    if config.get("ATTN_IMPL"):
        args.extend(["--attn_impl", config["ATTN_IMPL"]])
    if int(config.get("NPROC_PER_NODE") or "1") > 1:
        args.extend(["--deepspeed", "zero2"])
    return args


def write_command(path: Path, config: dict[str, str], args: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    quoted = " \\\n  ".join(shlex.quote(part) for part in args)
    nproc = shlex.quote(config.get("NPROC_PER_NODE") or "1")
    text = f"""#!/usr/bin/env bash
set -euo pipefail
cd {shlex.quote(str(ROOT))}
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="${{CUDA_VISIBLE_DEVICES:-0}}"
export NPROC_PER_NODE={nproc}
{quoted}
"""
    path.write_text(text, encoding="utf-8", newline="\n")


def _probe(module: str) -> str:
    try:
        __import__(module)
    except Exception as exc:
        return f"没有装好：{exc.__class__.__name__}: {exc}"
    return "可以导入"


def _gpus() -> list[str]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    completed = subprocess.run(
        [exe, "--query-gpu=name,memory.total,compute_cap", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    if completed.returncode != 0:
        return []
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def _memory_mib(line: str) -> int | None:
    for part in line.split(","):
        token = part.strip().split()
        if len(token) >= 2 and token[1].lower().startswith("mi"):
            try:
                return int(token[0])
            except ValueError:
                return None
    return None


def _cap_major(line: str) -> int | None:
    last = line.split(",")[-1].strip()
    try:
        return int(last.split(".", 1)[0])
    except ValueError:
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="检查训练环境并写出 swift 命令")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "sft_9b.env")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    config = merged_config(args.config)
    data_out = ROOT / config["DATA_OUT"]
    train_path = (data_out / "train.jsonl").resolve()
    val_path = (data_out / "val.jsonl").resolve()
    swift_args = build_swift_args(config, train_path, val_path)
    command_path = ROOT / "output" / "debug" / "train_command.sh"
    write_command(command_path, config, swift_args)

    print("训练前先记住这几件事：")
    for index, line in enumerate(EXPLAINS, start=1):
        print(f"{index}. {line}")
    print()
    print(f"配置 {args.config}")
    print(f"模型 {config['MODEL']}")
    print(f"输出 {config['OUTPUT_DIR']}")
    print(f"max_length {config['MAX_LENGTH']}  epochs {config['NUM_TRAIN_EPOCHS']}  max_steps {config.get('MAX_STEPS') or '（按 epoch）'}")
    print()
    for module in ("swift", "transformers", "peft", "fla", "causal_conv1d"):
        print(f"import {module}: {_probe(module)}")
    gpus = _gpus()
    if not gpus:
        print("没检测到 nvidia-smi。这台机器可以学流程，正式训练放到有 GPU 的内网 Linux。")
    for line in gpus:
        print(f"GPU {line}")
        memory = _memory_mib(line)
        major = _cap_major(line)
        if "9B" in config["MODEL"] and memory is not None and memory < 20000:
            print("显存低于约 20GB。9B bf16 LoRA 会很紧，改跑 bash scripts/train.sh --4b")
        if config.get("TORCH_DTYPE") == "bfloat16" and major is not None and major < 8:
            print("算力低于 8.0。把 TORCH_DTYPE 改成 float16 再试。")
    print()
    print(f"命令已写入 {command_path}")
    swift_ok = _probe("swift").startswith("可以")
    dry_run = args.dry_run or os.environ.get("DRY_RUN") == "1"
    if not swift_ok and not dry_run:
        print("还没有 swift，不能开始训练。先看 README 的安装段。")
        return 2
    if dry_run:
        print("这是试运行，没有启动训练。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
