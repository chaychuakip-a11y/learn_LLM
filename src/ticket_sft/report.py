"""生成一张离线 HTML。浏览器打开即可，不需要联网。"""

from __future__ import annotations

import argparse
import html
from pathlib import Path

from ticket_sft.audit import SampleAudit, audit_jsonl
from ticket_sft.paths import ROOT
from ticket_sft.plot_log import find_history, loss_svg, split_series


def build_report(audits: list[SampleAudit], output_dir: Path) -> str:
    history = find_history(output_dir) if output_dir.exists() else []
    train, evals = split_series(history)
    chart = loss_svg(train, evals)
    chart_block = chart or "<p>还没有训练曲线。冒烟训练之后，重新生成本页就能看到。</p>"
    cards = "\n".join(_card(audit) for audit in audits)
    failed = sum(1 for audit in audits if not audit.ok)
    return f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8"/>
<title>问题单训练样本</title>
<style>
  body {{ margin: 0; font: 16px/1.55 "Segoe UI", "Noto Sans SC", sans-serif; background: #f6f7f9; color: #1f2933; }}
  main {{ max-width: 920px; margin: 0 auto; padding: 28px 20px 64px; }}
  h1 {{ font-size: 28px; margin: 0 0 8px; }}
  h2 {{ font-size: 20px; margin: 28px 0 8px; }}
  .note, .card {{ background: white; border: 1px solid #e5e7eb; border-radius: 10px; padding: 14px 16px; }}
  .card {{ margin-top: 16px; }}
  .fail {{ border-color: #b45309; }}
  .tag {{ display: inline-block; padding: 1px 8px; border-radius: 999px; font-size: 13px; background: #e8eef7; }}
  .bad {{ background: #fde8d0; }}
  pre {{ white-space: pre-wrap; word-break: break-word; background: #f8fafc; padding: 10px; border-radius: 8px; }}
  table {{ border-collapse: collapse; width: 100%; }}
  td {{ border-top: 1px solid #e5e7eb; padding: 4px 6px; font-family: ui-monospace, Consolas, monospace; font-size: 14px; }}
  .learn {{ color: #1d4e89; font-weight: 650; }}
</style>
</head>
<body>
<main>
  <h1>问题单训练样本</h1>
  <p>{len(audits)} 条，其中 {failed} 条没通过检查。蓝字是模型要学的，其余只作为上下文。</p>
  <div class="note">
    <p>一条样本的顺序是：问题单 → 脚本 → 沙箱标准输出 → 结论。</p>
    <p>脚本和结论参与 loss。问题单、系统提示、标准输出不参与 loss。标准输出是证据，模型不该把它背下来。</p>
    <p>示例验证集只有 1 条。橙线只能说明流程没断，不能说明各个车厂语种已经学会。</p>
  </div>
  <h2>训练曲线</h2>
  <div class="note">{chart_block}</div>
  <h2>样本</h2>
  {cards}
</main>
</body>
</html>
"""


def _card(audit: SampleAudit) -> str:
    state = "通过" if audit.ok else "失败"
    tag = "tag" if audit.ok else "tag bad"
    rows = []
    for role, size in {**audit.context_chars, **audit.learn_chars}.items():
        kind = "要学" if role in audit.learn_chars else "上下文"
        css = "learn" if role in audit.learn_chars else ""
        rows.append(f"<tr><td class='{css}'>{kind}</td><td>{html.escape(role)}</td><td>{size} 字</td></tr>")
    finding_lines = "".join(
        f"<li>[{html.escape(item.level)}] {html.escape(item.message)}</li>" for item in audit.findings
    ) or "<li>没有问题</li>"
    css = "card" if audit.ok else "card fail"
    return f"""<article class="{css}" id="{html.escape(audit.case_id)}">
  <h2><span class="{tag}">{state}</span> {html.escape(audit.case_id)}</h2>
  <p>粗估 {audit.rough} token</p>
  <table>{''.join(rows)}</table>
  <h3>检查</h3>
  <ul>{finding_lines}</ul>
  <h3>脚本</h3>
  <pre>{html.escape(audit.code)}</pre>
  <h3>标准输出</h3>
  <pre>{html.escape(audit.stdout)}</pre>
  <h3>结论</h3>
  <pre>{html.escape(audit.conclusion)}</pre>
</article>
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="写出 output/debug/report.html")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--out", type=Path, default=ROOT / "output" / "debug" / "report.html")
    args = parser.parse_args(argv)
    paths = [ROOT / "data" / "processed" / "train.jsonl", ROOT / "data" / "processed" / "val.jsonl"]
    audits: list[SampleAudit] = []
    for path in paths:
        audits.extend(audit_jsonl(path, args.max_length))
    if not audits:
        print("没有样本。先运行 python -m ticket_sft.build_dataset")
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(build_report(audits, ROOT / "output"), encoding="utf-8")
    errors = sum(1 for audit in audits if not audit.ok)
    print(f"已写入 {args.out}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
