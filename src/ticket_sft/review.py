"""把人审后的唯一候选移入现有 cases/gold 布局，绝不自动把教师输出当金标。"""
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from ticket_sft.evidence import read_case, sha256, source_path
from ticket_sft.pipeline import write_json
from ticket_sft.rules import check_conclusion, loads_stdout
from ticket_sft.sandbox import run_code
from ticket_sft.protocol import validate_code


def approve(candidate, dataset, reviewer, group_id, split, note, demo=False):
    candidate, dataset = Path(candidate).resolve(), Path(dataset).resolve()
    if not reviewer.strip() or not note.strip() or not group_id.strip():
        raise ValueError('E_REVIEW: 必须填写审核人、分组和证据核对备注')
    if split not in {'train', 'val', 'test'}:
        raise ValueError('E_SPLIT: 只支持 train/val/test')
    result = json.loads((candidate / 'result.json').read_text(encoding='utf-8'))
    if result['status'] != 'pending_review':
        raise ValueError('E_REVIEW: 失败候选不能批准')
    source_dir = Path(result['case_dir'])
    case = read_case(source_dir, require_real=not demo)
    if not demo and group_id != case['group_id']:
        raise ValueError('E_GROUP: 审核分组必须与来源快照一致')
    if sha256(source_dir / 'case.json') != result['case_sha256']:
        raise ValueError('E_CASE_CHANGED: 源案例已变化，请重新采集')
    for name in ['analysis.py', 'conclusion.md', 'stdout.json', 'tool_result.json', 'transcript.json']:
        if sha256(candidate / name) != result.get('artifact_hashes', {}).get(name):
            raise ValueError('E_CANDIDATE_CHANGED: 候选产物变化，请重新采集 ' + name)
    code = validate_code((candidate / 'analysis.py').read_text(encoding='utf-8'))
    replay = run_code(code, source_dir)
    if not replay.ok or loads_stdout(replay.stdout) != loads_stdout((candidate / 'stdout.json').read_text(encoding='utf-8')):
        raise ValueError('E_REPLAY: 审核重放与候选证据不同')
    conclusion = (candidate / 'conclusion.md').read_text(encoding='utf-8')
    if any(f.level == 'error' for f in check_conclusion(loads_stdout(replay.stdout), conclusion)):
        raise ValueError('E_RULES: 规则检查失败')
    cid = case['case_id']
    target_case = dataset / 'cases' / cid
    target_gold = dataset / 'gold'
    if target_case.exists() or any((target_gold / (cid + suffix)).exists() for suffix in ['.py', '.md', '.meta.json']):
        raise ValueError('E_DUPLICATE: 此案例已有金标，禁止两个教师重复入库')
    target_case.mkdir(parents=True)
    target_gold.mkdir(parents=True, exist_ok=True)
    for source in case.get('sources', []):
        path = source_path(source_dir, source['path'])
        dest = target_case / source['path']
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
    case['group_id'] = group_id
    if demo:
        case['data_origin'] = 'synthetic'
    write_json(target_case / 'case.json', case)
    (target_gold / (cid + '.py')).write_text(code, encoding='utf-8')
    (target_gold / (cid + '.md')).write_text(conclusion, encoding='utf-8')
    write_json(target_gold / (cid + '.meta.json'), {
        'split': split, 'group_id': group_id, 'review_status': 'approved',
        'reviewer': reviewer, 'review_note': note, 'reviewed_at': datetime.now(timezone.utc).isoformat(),
        'data_origin': 'synthetic' if demo else 'real',
        'candidate': str(candidate), 'teacher': result['policy'],
        'case_sha256': sha256(target_case / 'case.json'),
        'code_sha256': sha256(target_gold / (cid + '.py')),
        'conclusion_sha256': sha256(target_gold / (cid + '.md'))})
    return cid
