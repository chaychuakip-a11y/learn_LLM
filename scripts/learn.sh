#!/usr/bin/env bash
# 第 0 课：不碰 GPU，把数据、沙箱和报告跑通。
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

echo "==== 自测 ===="
"$PY" -m ticket_sft.selftest

echo "==== 生成 jsonl ===="
"$PY" -m ticket_sft.build_dataset

echo "==== 看一条中文联系人单：哪一段要学 ===="
"$PY" -m ticket_sft.inspect --id zh_contact_004

echo "==== 写出报告 ===="
"$PY" -m ticket_sft.report

echo
echo "打开 output/debug/report.html"
echo "基础知识从第 1 课看起：python -m ticket_sft.basics"
echo "下一课在内网 Linux 上：bash scripts/train.sh --dry-run"
