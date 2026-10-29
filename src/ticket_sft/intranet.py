"""内网主入口。只有 teacher/analyze/train 的 --execute 会请求服务或训练。

无 --execute 时打印计划；demo 始终只重放仓库的合成案例，不伪装教师质量。
每个子命令只负责一个阶段，便于逐步学习和定位失败边界。
"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import time

from ticket_sft.dataset import build_reviewed
from ticket_sft.evidence import read_case, sha256, source_path
from ticket_sft.gateway import Client, load_endpoint
from ticket_sft.paths import ROOT
from ticket_sft.pipeline import diagnose, write_json
from ticket_sft.progress import format_bar
from ticket_sft.protocol import action_text
from ticket_sft.review import approve
from ticket_sft.runtime_log import RunLogger


def parser():
    p = argparse.ArgumentParser(description='内网多语种诊断：demo → teacher → approve → build → tokens → train → analyze')
    sub = p.add_subparsers(dest='command', required=True)
    demo = sub.add_parser('demo', help='完整离线闭环：4个合成案例，零API/零模型')
    demo.add_argument('--out', type=Path, required=True)
    doctor = sub.add_parser('doctor', help='只读查看运行环境，不安装依赖')
    doctor.add_argument('--hardware', action='store_true', help='额外调用 nvidia-smi')
    for name in ['teacher', 'analyze']:
        run = sub.add_parser(name, help='内网API诊断；默认只打印计划')
        run.add_argument('--config', type=Path, required=True)
        run.add_argument('--endpoint', required=True, help='配置内的 deepseek/glm/student 等名称')
        group = run.add_mutually_exclusive_group(required=True)
        group.add_argument('--case', type=Path)
        group.add_argument('--cases', type=Path, help='多个案例目录的父目录')
        run.add_argument('--out', type=Path, required=True)
        run.add_argument('--limit', type=int, default=0, help='0表示全部')
        run.add_argument('--sandbox-timeout', type=float, default=8)
        run.add_argument('--execute', action='store_true')
    review = sub.add_parser('approve', help='人读完候选证据后记录审核并导出金标')
    review.add_argument('--candidate', type=Path, required=True)
    review.add_argument('--dataset', type=Path, required=True)
    review.add_argument('--reviewer', required=True)
    review.add_argument('--group-id', required=True)
    review.add_argument('--split', required=True, choices=['train', 'val', 'test'])
    review.add_argument('--note', required=True, help='填写核对过的证据、保留的假设和缺口')
    curate = sub.add_parser('curate', help='人工修订脚本/结论后重新执行，生成新的待审核候选')
    curate.add_argument('--case', type=Path, required=True)
    curate.add_argument('--code', type=Path, required=True)
    curate.add_argument('--conclusion', type=Path, required=True)
    curate.add_argument('--out', type=Path, required=True)
    curate.add_argument('--source-candidate', type=Path, help='可选：保留原教师候选的来源路径')
    build = sub.add_parser('build', help='构建两个训练格式，并检查分组泄漏')
    build.add_argument('--dataset', type=Path, required=True)
    build.add_argument('--out', type=Path, required=True)
    select = sub.add_parser('select', help='按审核split复制案例及声明附件，不复制gold答案')
    select.add_argument('--dataset', type=Path, required=True)
    select.add_argument('--split', choices=['train', 'val', 'test'], required=True)
    select.add_argument('--out', type=Path, required=True)
    tokens = sub.add_parser('tokens', help='只加载本地tokenizer，检查真实长度和答案掩码')
    tokens.add_argument('--model', type=Path, required=True)
    tokens.add_argument('--data', type=Path, required=True, help='Gemma格式JSONL')
    tokens.add_argument('--max-length', type=int, default=2048)
    tokens.add_argument('--out', type=Path, required=True)
    train = sub.add_parser('train', help='四张V100的Gemma FP16 LoRA/ZeRO-3；默认打印命令')
    train.add_argument('--model', type=Path, required=True)
    train.add_argument('--data', type=Path, required=True, help='build输出内的gemma目录')
    train.add_argument('--out', type=Path, required=True)
    train.add_argument('--max-length', type=int, default=2048)
    train.add_argument('--smoke', action='store_true', help='最多20个更新步骤')
    train.add_argument('--allow-demo-smoke', action='store_true', help='只允许模拟数据冒烟，不用于正式训练')
    train.add_argument('--resume', type=Path)
    train.add_argument('--execute', action='store_true')
    watch = sub.add_parser('logs', help='终端查看逐步JSONL日志、错误和显存')
    watch.add_argument('--run', type=Path, required=True)
    watch.add_argument('--tail', type=int, default=12)
    watch.add_argument('--follow', action='store_true')
    package = sub.add_parser('deploy-plan', help='输出合并、量化、CPU服务命令，不执行')
    package.add_argument('--model', required=True)
    package.add_argument('--adapter', required=True)
    package.add_argument('--merged', required=True)
    package.add_argument('--llama-dir', required=True)
    return p


def demo(output):
    if output.exists():
        raise ValueError('E_OUTPUT_EXISTS: 每次演练用新目录，避免覆盖调试记录')
    splits = ['train', 'train', 'val', 'test']
    cases = sorted((ROOT / 'data/example/cases').iterdir())
    for index, (case_dir, split) in enumerate(zip(cases, splits), 1):
        cid = case_dir.name
        code = (ROOT / 'data/example/gold' / (cid + '.py')).read_text(encoding='utf-8')
        conclusion = (ROOT / 'data/example/gold' / (cid + '.md')).read_text(encoding='utf-8')
        replies = iter([action_text({'action': 'run_analysis', 'code': code}),
                        action_text({'action': 'final', 'conclusion': conclusion})])
        result = diagnose(case_dir, output / 'candidates' / cid, lambda _: next(replies), 'synthetic-replay')
        print(f"{format_bar(index, len(cases))} {cid} {result['status']} {result['duration_s']:.3f}s", flush=True)
        if result['status'] != 'pending_review':
            raise ValueError(result.get('error', 'demo failed'))
        approve(output / 'candidates' / cid, output / 'reviewed', 'demo-runner', cid,
                split, '合成示例演练，禁止作为真实效果或真实标签', demo=True)
    report = build_reviewed(output / 'reviewed', output / 'data', demo=True)
    print(json.dumps(report['counts'], ensure_ascii=False))
    print('完成：4个合成案例 / 8条决策。candidates看逐步输出，data看两种训练格式。')
    print('网络请求=0，模型加载=0，GPU训练=0。下一步：python -m ticket_sft.intranet logs --run ' + str(output))
    return 0


def doctor(hardware=False):
    print(f'Python {platform.python_version()} | {platform.platform()}')
    for name in ['torch', 'transformers', 'peft', 'accelerate', 'deepspeed']:
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            version = '未安装（纯Python demo仍可运行）'
        print(f'{name:14s} {version}')
    if hardware:
        r = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv'],
                           capture_output=True, text=True, timeout=20)
        print(r.stdout or r.stderr)
        if r.returncode:
            return 1
    print('以上只报告环境；V100算子、显存、FP16稳定性和CPU吞吐由真实冒烟验证。')
    return 0


def curate(a):
    if a.out.exists():
        raise ValueError('E_OUTPUT_EXISTS: 人工修订使用新的候选目录')
    replies = iter([action_text({'action': 'run_analysis', 'code': a.code.read_text(encoding='utf-8')}),
                    action_text({'action': 'final', 'conclusion': a.conclusion.read_text(encoding='utf-8')})])
    result = diagnose(a.case, a.out, lambda _: next(replies), 'human-curated')
    write_json(a.out / 'curation_source.json', {'source_candidate': str(a.source_candidate) if a.source_candidate else None,
                                              'code_sha256': sha256(a.code), 'conclusion_sha256': sha256(a.conclusion)})
    print(result['status'])
    return 0 if result['status'] == 'pending_review' else 1


def select_cases(a):
    if a.out.exists():
        raise ValueError('E_OUTPUT_EXISTS: 测试入口使用新目录')
    selected = []
    for path in sorted((a.dataset / 'gold').glob('*.meta.json')):
        meta = json.loads(path.read_text(encoding='utf-8'))
        if meta.get('split') != a.split:
            continue
        cid = path.name.removesuffix('.meta.json')
        case_dir = a.dataset / 'cases' / cid
        case = read_case(case_dir)
        if meta.get('review_status') != 'approved' or sha256(case_dir / 'case.json') != meta.get('case_sha256'):
            raise ValueError('E_REVIEW_STALE: 来源案例未审核或已变化 ' + cid)
        selected.append((cid, case_dir, case))
    if not selected:
        raise ValueError('E_EMPTY_SPLIT: 没有该集合案例')
    for cid, case_dir, case in selected:
        target = a.out / cid
        target.mkdir(parents=True)
        shutil.copyfile(case_dir / 'case.json', target / 'case.json')
        for source in case.get('sources', []):
            source_file = source_path(case_dir, source['path'])
            dest = target / source['path']
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_file, dest)
    print(f'{a.split}: {len(selected)}个案例；保留输入字节，不复制gold答案。')
    return 0


def run_api(a):
    endpoint = load_endpoint(a.config, a.endpoint)
    paths = [a.case] if a.case else sorted(p.parent for p in a.cases.glob('*/case.json'))
    if a.limit < 0:
        raise ValueError('E_LIMIT: limit不得小于0')
    if a.limit:
        paths = paths[:a.limit]
    if not paths:
        raise ValueError('E_CASES: 未找到案例')
    cases = [(p, read_case(p)) for p in paths]
    if len({c['case_id'] for _, c in cases}) != len(cases):
        raise ValueError('E_DUPLICATE: case_id重复')
    print(f"计划：{a.endpoint} / {endpoint['model']} / {len(cases)}个案例，每例最多2次API请求")
    if not a.execute:
        print('仅检查计划。填好配置与API环境变量后加 --execute。')
        return 0
    if a.out.exists():
        raise ValueError('E_OUTPUT_EXISTS: 使用新的批次目录；不会重发已完成请求')
    failures = 0
    for index, (path, case) in enumerate(cases, 1):
        directory = a.out / case['case_id']
        client = Client(endpoint, directory / 'api', case['case_id'])
        logger = RunLogger(directory / 'events.jsonl', 'diagnosis', secrets=(client.key,), run_id=client.log.run_id)
        result = diagnose(path, directory, client.chat, a.endpoint, a.sandbox_timeout, logger)
        failures += result['status'] == 'failed'
        print(f"{format_bar(index, len(cases))} {case['case_id']} {result['status']} {result['duration_s']:.2f}s", flush=True)
    write_json(a.out / 'batch_summary.json', {'total': len(cases), 'failed': failures,
                                             'pending_review': len(cases) - failures})
    return 1 if failures else 0


def token_check(a):
    if not a.model.is_dir():
        raise ValueError('E_MODEL_PATH: tokenizer需要现有本地模型目录')
    from transformers import AutoProcessor
    from ticket_sft.training_text import encode_example
    tokenizer = AutoProcessor.from_pretrained(str(a.model), local_files_only=True).tokenizer
    report = []
    for index, line in enumerate(a.data.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        encoded = encode_example(tokenizer, row['prompt'], row['completion'], a.max_length)
        target = [v for v in encoded['labels'] if v != -100]
        report.append({'line': index, 'total_tokens': len(encoded['input_ids']),
                       'target_tokens': len(target), 'target_preview': tokenizer.decode(target[:80])})
        print(f"line {index:5d} total={len(encoded['input_ids']):5d} target={len(target):4d}")
    if not report:
        raise ValueError('E_EMPTY_DATA')
    a.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(a.out, report)
    return 0


def train(a):
    report = json.loads((a.data / 'split_report.json').read_text(encoding='utf-8'))
    if report['demo_only'] and not (a.smoke and a.allow_demo_smoke):
        raise ValueError('E_DEMO_TRAIN: 模拟数据仅可 --smoke --allow-demo-smoke')
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4',
               '-m', 'ticket_sft.train_gemma4', '--model', str(a.model), '--data', str(a.data),
               '--output', str(a.out), '--max-length', str(a.max_length)]
    if a.smoke:
        command += ['--max-steps', '20', '--save-steps', '10', '--eval-steps', '10']
    if a.resume:
        command += ['--resume-from-checkpoint', str(a.resume)]
    print(shlex.join(command))
    print('FP16 | ZeRO-3 | LoRA rank16 | batch=1 × 4卡 × 累积8 = 32条决策/更新')
    if not a.execute:
        print('只显示命令。--execute 才加载模型；默认上下文2048，超长会报错，不截断。')
        return 0
    if not a.model.is_dir():
        raise ValueError('E_MODEL_PATH: 模型尚未放到本地目录')
    if a.out.exists() and not a.resume:
        raise ValueError('E_OUTPUT_EXISTS: 请使用新的训练目录或指定 --resume')
    if a.resume and not a.resume.is_dir():
        raise ValueError('E_CHECKPOINT: 恢复目录不存在')
    if a.resume and a.resume.resolve().parent != a.out.resolve():
        raise ValueError('E_CHECKPOINT: 从本次output目录下的checkpoint恢复，避免混用训练版本')
    a.out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'PYTHONUNBUFFERED': '1'})
    write_json(a.out / 'launch.json', {'command': command, 'data_report': report})
    with (a.out / 'console.log').open('a', encoding='utf-8') as log:
        child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, encoding='utf-8', errors='replace', env=env)
        try:
            for line in child.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
            return child.wait()
        except KeyboardInterrupt:
            child.terminate()
            child.wait()
            raise


def logs(directory, tail=12, follow=False):
    if tail <= 0 or not directory.is_dir():
        raise ValueError('E_LOG_PATH: 需要已存在的运行目录和正数tail')
    last = None
    try:
        while True:
            files = sorted(directory.rglob('events.jsonl')) + sorted(directory.rglob('rank_*.jsonl'))
            chunks = []
            for file in files:
                lines = file.read_text(encoding='utf-8').splitlines()[-tail:]
                chunks.append('\n[' + str(file.relative_to(directory)) + ']')
                for line in lines:
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        chunks.append('日志末行正在写入或损坏，保留原文件检查')
                        continue
                    metric = item.get('metrics', {})
                    details = {key: item[key] for key in ['case_id', 'request_id', 'latency_s', 'exit_code',
                               'stdout_chars', 'stderr_chars', 'usage', 'target_modules'] if key in item}
                    chunks.append(f"{item.get('elapsed_s', 0):8.2f}s {item.get('level', 'INFO'):5s} "
                                  f"{item.get('event')} step={item.get('step', '-')} "
                                  f"loss={metric.get('loss', '-')} VRAM_MiB={item.get('allocated_mib', '-')} "
                                  f"{item.get('message', item.get('status', ''))} "
                                  + (json.dumps(details, ensure_ascii=False) if details else ''))
            view = '\n'.join(chunks) or '尚未找到 events.jsonl / rank_*.jsonl'
            if view != last:
                print(view, flush=True)
                last = view
            if not follow:
                return 0
            time.sleep(1)
    except KeyboardInterrupt:
        return 0


def deploy_plan(a):
    merged = Path(a.merged)
    llama = Path(a.llama_dir)
    commands = [
        ['python', '-m', 'ticket_sft.merge_adapter', '--model', a.model, '--adapter', a.adapter, '--output', a.merged],
        ['python', str(llama / 'convert_hf_to_gguf.py'), a.merged, '--outtype', 'bf16', '--outfile', str(merged) + '.bf16.gguf'],
        [str(llama / 'build/bin/llama-quantize'), str(merged) + '.bf16.gguf', str(merged) + '.Q8_0.gguf', 'Q8_0'],
        [str(llama / 'build/bin/llama-server'), '-m', str(merged) + '.Q8_0.gguf', '--alias', 'asr-gemma',
         '-ngl', '0', '-c', '4096', '-t', '16', '-np', '1', '--jinja', '--host', '127.0.0.1', '--port', '8080'],
    ]
    for command in commands:
        print(shlex.join(command))
    print('仅输出命令。先比较合并/量化前后结果，再测CPU端完整请求耗时；CentOS7兼容性需目标机确认。')
    return 0


def main(argv=None):
    a = parser().parse_args(argv)
    try:
        if a.command == 'demo': return demo(a.out)
        if a.command == 'doctor': return doctor(a.hardware)
        if a.command == 'curate': return curate(a)
        if a.command == 'select': return select_cases(a)
        if a.command in {'teacher', 'analyze'}: return run_api(a)
        if a.command == 'approve':
            print(approve(a.candidate, a.dataset, a.reviewer, a.group_id, a.split, a.note))
            return 0
        if a.command == 'build':
            print(json.dumps(build_reviewed(a.dataset, a.out)['counts'], ensure_ascii=False))
            return 0
        if a.command == 'tokens': return token_check(a)
        if a.command == 'train': return train(a)
        if a.command == 'logs': return logs(a.run, a.tail, a.follow)
        if a.command == 'deploy-plan': return deploy_plan(a)
    except Exception as exc:
        print(f'[ERROR] {type(exc).__name__}: {exc}', file=sys.stderr)
        print('保留本次运行目录，先查 result.json / stderr.log / api/*.response.txt / events.jsonl。', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
