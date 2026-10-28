"""python -m ticket_sft 时打印学习顺序。"""

from ticket_sft import __version__


def main() -> int:
    print(
        f"""ticket-sft {__version__}

基础知识，一次一课：
  python -m ticket_sft.basics
  python -m ticket_sft.basics 1

第 0 课，不需要 GPU：
  bash scripts/learn.sh
Windows 上等价于：
  $env:PYTHONPATH = "src"
  python -m ticket_sft.selftest
  python -m ticket_sft.build_dataset
  python -m ticket_sft.inspect --id zh_contact_004
  python -m ticket_sft.report

然后打开 output/debug/report.html。

内网 Linux 上，环境装好之后：
  bash scripts/train.sh --dry-run
  bash scripts/train.sh --smoke
  bash scripts/train.sh

显存不够 24GB 时：
  bash scripts/train.sh --4b --dry-run

说明写在 README.md。
"""
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
