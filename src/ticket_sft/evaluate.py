"""按隔离测试集评估完整诊断；技术正确率只统计人工审核过的条目。"""
import argparse
import json
import math
from pathlib import Path

from ticket_sft.pipeline import write_json
from ticket_sft.evidence import sha256


def evaluate(dataset, run, out, assessments=None):
    report = json.loads((Path(dataset) / 'split_report.json').read_text(encoding='utf-8'))
    expected = [r for r in report['records'] if r['split'] == 'test']
    if not expected:
        raise ValueError('E_EVAL: 测试集为空')
    manual = {}
    if assessments:
        for row in json.loads(Path(assessments).read_text(encoding='utf-8')):
            cid = row['case_id']
            if cid in manual or cid not in {r['case_id'] for r in expected}:
                raise ValueError('E_EVAL_ID: 人工评测有重复或非测试case_id')
            for key in ['resource_correct', 'unsupported_root_cause', 'missing_evidence_handled']:
                if row.get(key) is not None and type(row[key]) is not bool:
                    raise ValueError('E_ASSESSMENT: 人工栏只接受true/false/null')
            if any(row.get(k) is not None for k in ['resource_correct', 'unsupported_root_cause', 'missing_evidence_handled']) and not row.get('reviewer'):
                raise ValueError('E_ASSESSMENT: 已填写的人工结果必须记录reviewer')
            manual[cid] = row
    rows, durations = [], []
    for record in expected:
        cid = record['case_id']
        result_file = Path(run) / cid / 'result.json'
        result = json.loads(result_file.read_text(encoding='utf-8')) if result_file.is_file() else {}
        if result and result.get('case_id') != cid:
            raise ValueError('E_EVAL_ID: 预测case_id不匹配')
        if result and result.get('case_sha256') != record['case_sha256']:
            raise ValueError('E_EVAL_SNAPSHOT: 预测输入不是该测试集冻结快照')
        complete = result.get('status') == 'pending_review'
        if complete:
            for name in ['analysis.py', 'conclusion.md', 'stdout.json', 'tool_result.json', 'transcript.json']:
                path = Path(run) / cid / name
                if not path.is_file() or sha256(path) != result.get('artifact_hashes', {}).get(name):
                    raise ValueError('E_EVAL_ARTIFACT: 预测产物缺失或变化 ' + cid + ':' + name)
        duration = result.get('duration_s')
        if isinstance(duration, (int, float)) and math.isfinite(duration) and duration >= 0:
            durations.append(duration)
        expert = manual.get(cid, {})
        row = {'case_id': cid, 'language': record['language'], 'prediction_present': bool(result),
               'pipeline_passed': complete, 'duration_s': duration,
               'resource_correct': expert.get('resource_correct'),
               'unsupported_root_cause': expert.get('unsupported_root_cause'),
               'missing_evidence_handled': expert.get('missing_evidence_handled'),
               'reviewer': expert.get('reviewer', ''), 'note': expert.get('note', '')}
        if not complete and any(row[k] is not None for k in ['resource_correct', 'unsupported_root_cause', 'missing_evidence_handled']):
            raise ValueError('E_ASSESSMENT: 未完成预测不能标成已做语义评测')
        rows.append(row)
    def summary(group):
        result = {'total': len(group), 'pipeline_passed': sum(r['pipeline_passed'] for r in group)}
        for metric in ['resource_correct', 'unsupported_root_cause', 'missing_evidence_handled']:
            reviewed = [r[metric] for r in group if r[metric] is not None]
            result[metric] = {'reviewed': len(reviewed), 'rate': sum(reviewed)/len(reviewed) if reviewed else None}
        return result
    output = {'demo_only': report['demo_only'], 'overall': summary(rows),
              'by_language': {lang: summary([r for r in rows if r['language'] == lang]) for lang in sorted({r['language'] for r in rows})},
              'p95_duration_s': sorted(durations)[math.ceil(len(durations)*0.95)-1] if durations else None,
              'latency_observations': len(durations),
              'note': 'pipeline_passed仅表示流程与有限规则通过；不等于资源归因正确。p95需结合样本量。'}
    out = Path(out)
    if out.exists():
        raise ValueError('E_OUTPUT_EXISTS: 评测结果请存新版本目录')
    out.mkdir(parents=True)
    write_json(out / 'metrics.json', output)
    write_json(out / 'review_sheet.json', rows)
    return output


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=Path, required=True, help='含split_report.json的数据版本目录')
    p.add_argument('--run', type=Path, required=True, help='analyze输出的批次目录')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--assessments', type=Path)
    a = p.parse_args(argv)
    result = evaluate(a.dataset, a.run, a.out, a.assessments)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
