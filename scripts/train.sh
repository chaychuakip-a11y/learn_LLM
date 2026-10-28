#!/usr/bin/env bash
# 在内网 Linux GPU 上训练。先 --dry-run，再 --smoke，最后去掉参数正式训。
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

CONFIG_REL="${CONFIG:-configs/sft_9b.env}"
DRY=0
SMOKE=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY=1 ;;
    --smoke) SMOKE=1 ;;
    --4b) CONFIG_REL="configs/sft_4b.env" ;;
    *)
      echo "未知参数: $arg"
      echo "可用：--dry-run  --smoke  --4b"
      exit 2
      ;;
  esac
done

CONFIG="$ROOT/$CONFIG_REL"
set -a
# shellcheck disable=SC1090
source "$CONFIG"
set +a
if [[ "$SMOKE" == 1 ]]; then
  export MAX_STEPS=20
  export SAVE_STEPS=10
  export EVAL_STEPS=10
  export SMOKE=1
fi
if [[ "$DRY" == 1 ]]; then
  export DRY_RUN=1
fi

"$PY" -m ticket_sft.build_dataset \
  --cases "$DATA_CASES" \
  --gold "$DATA_GOLD" \
  --out "$DATA_OUT" \
  --max-length "$MAX_LENGTH"
"$PY" -m ticket_sft.check_data \
  "$DATA_OUT/train.jsonl" \
  "$DATA_OUT/val.jsonl" \
  --max-length "$MAX_LENGTH"
"$PY" -m ticket_sft.preflight --config "$CONFIG"
if [[ "$DRY" == 1 ]]; then
  echo "试运行结束，没有启动 GPU 训练。"
  exit 0
fi
mkdir -p "$OUTPUT_DIR"
bash "$ROOT/output/debug/train_command.sh" 2>&1 | tee -a "$OUTPUT_DIR/train.log"
