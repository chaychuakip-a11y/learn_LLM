#!/usr/bin/env bash
# 无模型、无网络的回归验证；不会安装依赖。
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
"$PY" -m ticket_sft.selftest
"$PY" -m unittest discover -s tests -v
