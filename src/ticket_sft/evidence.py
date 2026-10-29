"""核验案例和本地来源文件；哈希核验不等于内容或根因正确。"""
import hashlib
import json
import re
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_case(directory, require_real=False):
    directory = Path(directory).resolve()
    case = json.loads((directory / 'case.json').read_text(encoding='utf-8'))
    cid = case.get('case_id', '')
    if not isinstance(cid, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,119}', cid):
        raise ValueError('E_CASE_ID: case_id 需为字母数字下划线等安全 ID')
    for field in ['oem', 'language', 'runtime', 'score_type', 'lexicon', 'utterances']:
        if field not in case:
            raise ValueError('E_CASE: 缺少 ' + field)
    if not isinstance(case['utterances'], list) or not case['utterances']:
        raise ValueError('E_CASE: utterances 不能为空')
    for utt in case['utterances']:
        if any(k not in utt for k in ['utt_id', 'lab', 'rec']):
            raise ValueError('E_CASE: 每句需要 utt_id/lab/rec')
        if any(not isinstance(utt[k], str) for k in ['utt_id', 'lab', 'rec']):
            raise ValueError('E_CASE: utt_id/lab/rec必须是字符串，空REC用空字符串')
    if len({u['utt_id'] for u in case['utterances']}) != len(case['utterances']):
        raise ValueError('E_CASE: utt_id重复')
    if require_real:
        if case.get('data_origin') != 'real' or not case.get('group_id'):
            raise ValueError('E_REAL: 正式数据需要 data_origin=real 和独立案例 group_id')
        if not case.get('sources') or not case.get('audio_fingerprints'):
            raise ValueError('E_PROVENANCE: 正式数据需要 sources 和 audio_fingerprints')
        if any(not re.fullmatch(r'[0-9a-f]{64}', x) for x in case['audio_fingerprints']):
            raise ValueError('E_AUDIO_HASH: 原音频指纹必须是 SHA256；转换副本沿用原音频关联')
        if not set(case['audio_fingerprints']) <= {s.get('sha256') for s in case['sources']}:
            raise ValueError('E_AUDIO_HASH: 原音频必须在来源清单中登记')
    for source in case.get('sources', []):
        file = source_path(directory, source['path'])
        if sha256(file) != source.get('sha256'):
            raise ValueError('E_SOURCE_HASH: 来源文件变化 ' + source['path'])
    return case


def source_path(directory, relative):
    directory = Path(directory).resolve()
    path = Path(relative)
    if path.is_absolute() or not relative or '..' in path.parts:
        raise ValueError('E_SOURCE_PATH: 来源文件必须在案例目录内')
    resolved = (directory / path).resolve()
    if directory not in resolved.parents or not resolved.is_file():
        raise ValueError('E_SOURCE_PATH: 来源文件不存在或越出案例目录')
    return resolved
