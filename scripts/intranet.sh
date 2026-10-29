#!/usr/bin/env bash
# 不安装依赖、不下载模型。兼容仓库原有源码运行方式。
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
exec "$PY" -m ticket_sft.intranet "$@"
