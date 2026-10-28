"""run_analysis 的工具定义。训练样本和推理循环共用这一份。"""

from __future__ import annotations

import json

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_analysis",
            "description": "在只读沙箱里执行 Python。案例位于 CASE_DIR/case.json，可 import analysis。把证据 JSON 打印到标准输出。",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "分析脚本全文",
                    }
                },
                "required": ["code"],
            },
        },
    }
]


def tools_string() -> str:
    return json.dumps(TOOLS, ensure_ascii=False)
