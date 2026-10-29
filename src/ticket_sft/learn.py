"""终端跟学目录：阅读一章，跑一个实际命令，再检查产物。"""
import argparse
import json
from ticket_sft.paths import ROOT


def main(argv=None):
    p = argparse.ArgumentParser(description='从0到内网诊断模型的项目路径')
    p.add_argument('step', nargs='?', type=int)
    p.add_argument('--done', action='store_true', help='完成练习后记录本地进度')
    a = p.parse_args(argv)
    chapters = sorted((ROOT / 'docs' / 'steps').glob('[0-9][0-9]-*.md'))
    progress_file = ROOT / 'output' / 'learning_progress.json'
    progress = json.loads(progress_file.read_text(encoding='utf-8')) if progress_file.is_file() else {}
    if a.step is None:
        print('内网诊断LLM：从一个可运行小例子逐步做到训练和部署\n')
        for index, path in enumerate(chapters):
            title = path.read_text(encoding='utf-8').splitlines()[0].lstrip('# ')
            print(f"[{'x' if progress.get(str(index)) else ' '}] {title}")
        print('\n查看：python -m ticket_sft learn 0\n记录完成：python -m ticket_sft learn 0 --done')
        return 0
    if not 0 <= a.step < len(chapters):
        p.error(f'步骤范围0..{len(chapters)-1}')
    if a.done:
        progress[str(a.step)] = True
        progress_file.parent.mkdir(parents=True, exist_ok=True)
        progress_file.write_text(json.dumps(progress, indent=2), encoding='utf-8')
        print(f'已记录第{a.step}步学习完成；这不是GPU或业务效果验收。')
    else:
        print(chapters[a.step].read_text(encoding='utf-8'))
    return 0
