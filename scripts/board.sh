#!/usr/bin/env bash
# 训练开始后另开一个终端，看 TensorBoard。没有它时用 plot_log 出 SVG。
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
PORT="${1:-6006}"
if ! "$PY" -c "import tensorboard" >/dev/null 2>&1; then
  echo "还没有 tensorboard。可以改看静态图："
  echo "  $PY -m ticket_sft.plot_log --output output"
  exit 1
fi
echo "浏览器打开 http://127.0.0.1:${PORT}"
exec "$PY" -m tensorboard.main --logdir "$ROOT/output" --bind_all --port "$PORT"
