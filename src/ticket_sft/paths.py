"""仓库根目录。始终按源码树定位，不依赖 pip install。"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
SYSTEM_PROMPT_PATH = ROOT / "prompts" / "system.md"

USER_PREAMBLE = "请分析下面这张问题单。先写分析脚本，再根据脚本输出作答。\n\n"

HEADINGS = ("口径", "脚本在查什么", "结论", "待核实", "正式门禁")
