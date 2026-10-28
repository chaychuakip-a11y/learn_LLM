"""把 trainer_state.json 或 logging.jsonl 画成一张 loss 图。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ticket_sft.paths import ROOT


def find_history(output_dir: Path) -> list[dict]:
    states = sorted(output_dir.rglob("trainer_state.json"), key=lambda path: path.stat().st_mtime)
    if states:
        payload = json.loads(states[-1].read_text(encoding="utf-8"))
        history = payload.get("log_history")
        if isinstance(history, list):
            return [row for row in history if isinstance(row, dict)]
    rows: list[dict] = []
    for path in sorted(output_dir.rglob("logging.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                rows.append(item)
    return rows


def split_series(history: list[dict]) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    train: list[tuple[int, float]] = []
    evals: list[tuple[int, float]] = []
    for row in history:
        step = row.get("step", row.get("global_step"))
        if step is None:
            continue
        step_no = int(step)
        if "loss" in row:
            train.append((step_no, float(row["loss"])))
        if "eval_loss" in row:
            evals.append((step_no, float(row["eval_loss"])))
    return train, evals


def loss_svg(train: list[tuple[int, float]], evals: list[tuple[int, float]]) -> str:
    if not train and not evals:
        return ""
    width, height = 760, 320
    left, right, top, bottom = 56, 16, 28, 36
    points = train + evals
    xs = [item[0] for item in points]
    ys = [item[1] for item in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    if min_x == max_x:
        max_x += 1
    if min_y == max_y:
        min_y -= 1
        max_y += 1
    inner_w = width - left - right
    inner_h = height - top - bottom

    def xy(step: int, loss: float) -> tuple[float, float]:
        x = left + (step - min_x) / (max_x - min_x) * inner_w
        y = top + (max_y - loss) / (max_y - min_y) * inner_h
        return x, y

    def polyline(series: list[tuple[int, float]], color: str) -> str:
        if not series:
            return ""
        if len(series) == 1:
            x, y = xy(*series[0])
            return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>'
        coords = " ".join(f"{xy(step, loss)[0]:.1f},{xy(step, loss)[1]:.1f}" for step, loss in series)
        return f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{coords}"/>'

    y_ticks = []
    for index in range(5):
        value = min_y + (max_y - min_y) * index / 4
        y = top + (max_y - value) / (max_y - min_y) * inner_h
        y_ticks.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" stroke="#e5e7eb"/>'
            f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="12" fill="#374151">{value:.2f}</text>'
        )
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="{left}" y="18" font-size="14" fill="#111827">训练 loss（蓝）和验证 loss（橙）</text>
  {''.join(y_ticks)}
  {polyline(train, "#1d4e89")}
  {polyline(evals, "#b45309")}
  <text x="{left}" y="{height - 8}" font-size="12" fill="#374151">step {min_x} → {max_x}</text>
</svg>
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="从训练日志画出 loss")
    parser.add_argument("--output", type=Path, default=ROOT / "output")
    parser.add_argument("--svg", type=Path, default=ROOT / "output" / "debug" / "loss.svg")
    args = parser.parse_args(argv)
    if not args.output.exists():
        print(f"还没有 {args.output}。先跑冒烟训练。")
        return 1
    history = find_history(args.output)
    train, evals = split_series(history)
    svg = loss_svg(train, evals)
    if not svg:
        print("日志里还没有 loss。确认训练已经打出 step，或等第一个 save_steps。")
        return 1
    args.svg.parent.mkdir(parents=True, exist_ok=True)
    args.svg.write_text(svg, encoding="utf-8")
    print(f"已写入 {args.svg}  训练点 {len(train)}  验证点 {len(evals)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
