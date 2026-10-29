"""审核数据 → 原仓库 swift 格式 + Gemma 两步格式，分组/音频不跨集合。"""
import json
from pathlib import Path

from ticket_sft.build_dataset import _one
from ticket_sft.evidence import read_case, sha256
from ticket_sft.paths import SYSTEM_PROMPT_PATH
from ticket_sft.pipeline import write_json
from ticket_sft.protocol import initial_messages, action_text, tool_message, validate_code


def decisions(case, record):
    """每条案例两条决策；stdout 只作为上下文，不计算目标 loss。"""
    messages = record['messages']
    call = json.loads(messages[2]['content'])
    code_action = action_text({'action': 'run_analysis', 'code': call['arguments']['code']})
    prompt = initial_messages(case)
    final_prompt = prompt + [{'role': 'assistant', 'content': code_action},
                             tool_message(json.loads(messages[3]['content']))]
    return [
        {'prompt': prompt, 'completion': [{'role': 'assistant', 'content': code_action}]},
        {'prompt': final_prompt, 'completion': [{'role': 'assistant', 'content': action_text(
            {'action': 'final', 'conclusion': messages[4]['content']})}]},
    ]


def build_reviewed(dataset, out, demo=False):
    dataset, out = Path(dataset), Path(out)
    if out.exists():
        raise ValueError('E_OUTPUT_EXISTS: 数据版本目录已存在，请使用新目录')
    native, gemma = ({s: [] for s in ['train', 'val', 'test']} for _ in range(2))
    groups, audio_splits, case_hash_splits = {}, {}, {}
    records = []
    system = SYSTEM_PROMPT_PATH.read_text(encoding='utf-8').strip()
    scripts = sorted((dataset / 'gold').glob('*.py'))
    if not scripts:
        raise ValueError('E_DATA: 没有审核后的脚本')
    for script in scripts:
        cid = script.stem
        case_dir = dataset / 'cases' / cid
        case = read_case(case_dir, require_real=not demo)
        meta = json.loads(script.with_suffix('.meta.json').read_text(encoding='utf-8'))
        if meta.get('review_status') != 'approved' or not meta.get('reviewer') or not meta.get('review_note'):
            raise ValueError('E_UNREVIEWED: 未审核 ' + cid)
        if not demo and meta.get('data_origin') != 'real':
            raise ValueError('E_SYNTHETIC: 模拟数据禁止进入真实训练 ' + cid)
        for key, path in [('case_sha256', case_dir / 'case.json'), ('code_sha256', script),
                          ('conclusion_sha256', script.with_suffix('.md'))]:
            if meta.get(key) != sha256(path):
                raise ValueError('E_REVIEW_STALE: 审核后内容变化 ' + cid + ':' + key)
        split, group = meta.get('split'), meta.get('group_id')
        if split not in native or not group or group != case.get('group_id'):
            raise ValueError('E_SPLIT: 缺少或不一致的 split/group_id ' + cid)
        # 哈希不同的转码副本也须由业务 group_id/原音频指纹关联。
        for bucket, keys in [(groups, [group]), (audio_splits, case.get('audio_fingerprints', [])),
                             (case_hash_splits, [meta['case_sha256']])]:
            for key in keys:
                if key in bucket and bucket[key] != split:
                    raise ValueError('E_LEAKAGE: 同组/同音频跨集合 ' + cid)
                bucket[key] = split
        validate_code(script.read_text(encoding='utf-8'))
        record, findings = _one(cid, dataset / 'cases', dataset / 'gold', system, 1_000_000_000)
        if record is None:
            raise ValueError('E_REPLAY: ' + '; '.join(f.message for f in findings))
        native[split].append(record)
        gemma[split].extend(decisions(case, record))
        records.append({'case_id': cid, 'group_id': group, 'split': split, 'language': case['language'],
                        'case_sha256': meta['case_sha256'], 'teacher': meta.get('teacher')})
    if any(not native[s] for s in native):
        raise ValueError('E_EMPTY_SPLIT: train/val/test 均须有独立案例；补数据，勿搬动已用测试组')
    # 验证完成后才写盘，避免失败时继续使用旧训练文件。
    for kind, grouped in [('swift', native), ('gemma', gemma)]:
        directory = out / kind
        directory.mkdir(parents=True)
        for split, rows in grouped.items():
            name = 'valid' if kind == 'gemma' and split == 'val' else split
            (directory / (name + '.jsonl')).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    report = {'demo_only': demo, 'counts': {s: len(native[s]) for s in native},
              'decision_counts': {s: len(gemma[s]) for s in gemma}, 'records': records,
              'group_splits': groups, 'audio_splits': audio_splits,
              'note': 'train/val/test已隔离；真实token上限在本地tokenizer预检和训练时严格检查。'}
    write_json(out / 'split_report.json', report)
    write_json(out / 'gemma' / 'split_report.json', report)
    return report
