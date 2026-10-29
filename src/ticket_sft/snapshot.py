"""冻结结构化案例和原始附件；不猜测私有dump字段，不改变标注口径。"""
import argparse
import json
from pathlib import Path
import shutil
from ticket_sft.evidence import source_path, sha256, read_case
from ticket_sft.pipeline import write_json


def snapshot(case_dir, out, sources, original_audio, group_id, origin):
    case_dir, out = Path(case_dir).resolve(), Path(out).resolve()
    if out.exists():
        raise ValueError('E_OUTPUT_EXISTS: 快照须使用新目录')
    if not sources or not original_audio or not group_id:
        raise ValueError('E_SNAPSHOT: 需要来源文件、原音频和group_id')
    case = json.loads((case_dir / 'case.json').read_text(encoding='utf-8'))
    paths = sorted(set(sources + original_audio))
    if any(Path(name).as_posix() == 'case.json' for name in paths):
        raise ValueError('E_SOURCE: case.json不能成为自身附件')
    resolved = {name: source_path(case_dir, name) for name in paths}
    case.update({'data_origin': origin, 'group_id': group_id,
                 'sources': [{'path': name, 'sha256': sha256(path)} for name, path in resolved.items()],
                 'audio_fingerprints': sorted({sha256(resolved[name]) for name in original_audio})})
    out.mkdir(parents=True)
    for name, path in resolved.items():
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    write_json(out / 'case.json', case)
    read_case(out, require_real=origin == 'real')
    return case


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case-dir', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--source', action='append', required=True, help='案例目录内相对路径，可多次传入')
    p.add_argument('--original-audio', action='append', required=True, help='原始音频相对路径，可多次传入')
    p.add_argument('--group-id', required=True)
    p.add_argument('--origin', choices=['real', 'synthetic'], required=True)
    a = p.parse_args(argv)
    case = snapshot(a.case_dir, a.out, a.source, a.original_audio, a.group_id, a.origin)
    print(f"{case['case_id']} sources={len(case['sources'])} audio={len(case['audio_fingerprints'])} group={case['group_id']}")
    print('已保存快照。文件哈希只证明来源一致，不证明标注或根因正确。')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
