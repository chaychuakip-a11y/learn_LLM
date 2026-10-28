# 由 learn.sh / train.sh source。不要直接执行。
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if command -v python3 >/dev/null 2>&1; then
  PY=python3
else
  PY=python
fi
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PYTHONUTF8=1
