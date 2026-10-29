"""运行：PYTHONPATH=src python -m unittest discover -s tests -v。

不依赖模型/GPU/网关；网络响应通过mock注入。所有业务样本都是仓库的合成例子。
"""
import copy
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from ticket_sft.dataset import build_reviewed, decisions
from ticket_sft.evidence import read_case, sha256, source_path
from ticket_sft.evaluate import evaluate
from ticket_sft.gateway import Client, GatewayError, load_endpoint, NoRedirect
from ticket_sft.lab import train_bigram
from ticket_sft.paths import ROOT
from ticket_sft.pipeline import diagnose, write_json
from ticket_sft.protocol import initial_messages, parse_action, action_text, tool_message
from ticket_sft.review import approve
from ticket_sft.sandbox import run_code
from ticket_sft.snapshot import snapshot
from ticket_sft.training_text import encode_example


class ToyTokenizer:
    def apply_chat_template(self, messages, add_generation_prompt=False, **kwargs):
        text = '<bos>' + ''.join('<' + m['role'] + '>' + m['content'] + '<end>' for m in messages)
        return text + ('<assistant><empty_thought>' if add_generation_prompt else '')

    def encode(self, text, **kwargs):
        return [ord(c) for c in text]


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.case_dir = ROOT / 'data/example/cases/de_nav_002'
        self.code = (ROOT / 'data/example/gold/de_nav_002.py').read_text(encoding='utf-8')
        self.conclusion = (ROOT / 'data/example/gold/de_nav_002.md').read_text(encoding='utf-8')

    def tearDown(self):
        self.temp.cleanup()

    def replay(self, directory=None, output=None):
        replies = iter([action_text({'action': 'run_analysis', 'code': self.code}),
                        action_text({'action': 'final', 'conclusion': self.conclusion})])
        return diagnose(directory or self.case_dir, output or self.root / 'candidate', lambda _: next(replies), 'mock')

    def real_fixture(self, cid='test_1', group='g1'):
        # 仅为测试“真实数据模式”的schema约束，不是实际业务材料。
        directory = self.root / cid
        directory.mkdir()
        case = json.loads((self.case_dir / 'case.json').read_text(encoding='utf-8'))
        case.update({'case_id': cid, 'group_id': group, 'data_origin': 'real'})
        raw = directory / 'raw.log'
        raw.write_text('synthetic test fixture ' + cid, encoding='utf-8')
        case['sources'] = [{'path': 'raw.log', 'sha256': sha256(raw)}]
        case['audio_fingerprints'] = [sha256(raw)]
        write_json(directory / 'case.json', case)
        return directory

    def reviewed_dataset(self):
        for index, split in enumerate(['train', 'val', 'test']):
            directory = self.real_fixture('case_' + str(index), 'group_' + str(index))
            candidate = self.root / ('candidate_' + str(index))
            result = self.replay(directory, candidate)
            self.assertEqual(result['status'], 'pending_review')
            approve(candidate, self.root / 'dataset', 'tester', 'group_' + str(index), split, 'fixture review')
        return self.root / 'dataset'

    def test_two_step_loop_and_training_prefix_match(self):
        self.assertEqual(self.replay()['status'], 'pending_review')
        transcript = json.loads((self.root / 'candidate/transcript.json').read_text(encoding='utf-8'))
        self.assertEqual(transcript[:2], initial_messages(read_case(self.case_dir)))
        result = json.loads((self.root / 'candidate/tool_result.json').read_text(encoding='utf-8'))
        self.assertEqual(transcript[3], tool_message(result))
        self.assertEqual(len(transcript), 5)

    def test_final_without_evidence_fails(self):
        result = diagnose(self.case_dir, self.root / 'c', lambda _: action_text({'action': 'final', 'conclusion': 'guess'}), 'mock')
        self.assertEqual(result['status'], 'failed')
        self.assertIn('E_SEQUENCE', result['error'])

    def test_wrong_case_stdout_rejected(self):
        result = diagnose(self.case_dir, self.root / 'c', lambda _: action_text({'action': 'run_analysis', 'code': 'print(\'{"case_id":"other"}\')'}), 'mock')
        self.assertIn('E_IDENTITY', result['error'])

    def test_unknown_action_and_dangerous_code(self):
        for text in ['{"action":"shell","code":"ls"}',
                     '{"action":"run_analysis","code":"import os"}',
                     '{"action":"run_analysis","code":"import analysis\\nanalysis.os.remove(1)"}',
                     '{"action":"run_analysis","code":"exec(\\"x\\")"}']:
            with self.assertRaises(ValueError): parse_action(text)

    def test_tool_failure_does_not_request_conclusion(self):
        calls = []
        def chat(messages):
            calls.append(messages)
            return action_text({'action': 'run_analysis', 'code': 'raise ValueError("broken")'})
        result = diagnose(self.case_dir, self.root / 'c', chat, 'mock')
        self.assertEqual(len(calls), 1)
        self.assertIn('E_TOOL_FAILED', result['error'])
        self.assertIn('broken', (self.root / 'c/stderr.log').read_text(encoding='utf-8'))

    def test_human_review_required_and_tamper_rejected(self):
        self.replay()
        with self.assertRaisesRegex(ValueError, 'E_REVIEW'):
            approve(self.root / 'candidate', self.root / 'ds', '', 'g', 'train', '', demo=True)
        (self.root / 'candidate/conclusion.md').write_text('edited', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'E_CANDIDATE_CHANGED'):
            approve(self.root / 'candidate', self.root / 'ds', 'person', 'g', 'train', 'checked', demo=True)

    def test_synthetic_not_promoted_to_real(self):
        self.replay()
        with self.assertRaisesRegex(ValueError, 'E_REAL'):
            approve(self.root / 'candidate', self.root / 'ds', 'person', 'g', 'train', 'checked')

    def test_review_build_and_evaluate_missing_prediction(self):
        dataset = self.reviewed_dataset()
        report = build_reviewed(dataset, self.root / 'built')
        self.assertEqual(report['counts'], {'train': 1, 'val': 1, 'test': 1})
        self.assertEqual(report['decision_counts']['train'], 2)
        rows = [json.loads(x) for x in (self.root / 'built/gemma/train.jsonl').read_text(encoding='utf-8').splitlines()]
        self.assertEqual(len(rows[0]['prompt']), 2)
        self.assertEqual(len(rows[1]['prompt']), 4)
        metrics = evaluate(self.root / 'built', self.root / 'missing', self.root / 'eval')
        self.assertEqual(metrics['overall']['pipeline_passed'], 0)
        self.assertIsNone(metrics['overall']['resource_correct']['rate'])

    def test_same_audio_across_split_rejected(self):
        dataset = self.reviewed_dataset()
        case_path = dataset / 'cases/case_1/case.json'
        case = json.loads(case_path.read_text(encoding='utf-8'))
        first = json.loads((dataset / 'cases/case_0/case.json').read_text(encoding='utf-8'))
        case['audio_fingerprints'] = first['audio_fingerprints']
        shutil.copyfile(dataset / 'cases/case_0/raw.log', dataset / 'cases/case_1/raw.log')
        case['sources'] = copy.deepcopy(first['sources'])
        write_json(case_path, case)
        meta_path = dataset / 'gold/case_1.meta.json'
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        meta['case_sha256'] = sha256(case_path)
        write_json(meta_path, meta)
        with self.assertRaisesRegex(ValueError, 'E_LEAKAGE'):
            build_reviewed(dataset, self.root / 'built')
        self.assertFalse((self.root / 'built').exists())

    def test_evaluation_requires_same_input_and_records_manual_coverage(self):
        dataset = self.reviewed_dataset()
        build_reviewed(dataset, self.root / 'built')
        self.replay(dataset / 'cases/case_2', self.root / 'prediction/case_2')
        metrics = evaluate(self.root / 'built', self.root / 'prediction', self.root / 'eval1')
        self.assertEqual(metrics['overall']['pipeline_passed'], 1)
        self.assertEqual(metrics['overall']['resource_correct']['reviewed'], 0)
        write_json(self.root / 'review.json', [{'case_id': 'case_2', 'resource_correct': True, 'reviewer': 'expert'}])
        reviewed = evaluate(self.root / 'built', self.root / 'prediction', self.root / 'eval2', self.root / 'review.json')
        self.assertEqual(reviewed['overall']['resource_correct']['rate'], 1)
        result_path = self.root / 'prediction/case_2/result.json'
        result = json.loads(result_path.read_text(encoding='utf-8'))
        result['case_sha256'] = '0' * 64
        write_json(result_path, result)
        with self.assertRaisesRegex(ValueError, 'E_EVAL_SNAPSHOT'):
            evaluate(self.root / 'built', self.root / 'prediction', self.root / 'eval3')

    def test_select_does_not_copy_gold_answers(self):
        from types import SimpleNamespace
        from ticket_sft.intranet import select_cases
        dataset = self.reviewed_dataset()
        select_cases(SimpleNamespace(dataset=dataset, split='test', out=self.root / 'selected'))
        self.assertEqual(sha256(self.root / 'selected/case_2/case.json'), sha256(dataset / 'cases/case_2/case.json'))
        self.assertFalse(any((self.root / 'selected').rglob('*.py')))
        self.assertEqual(len(list((self.root / 'selected').glob('*/case.json'))), 1)

    def test_empty_dump_is_missing_evidence(self):
        from ticket_sft.analysis import base_evidence
        case = read_case(self.case_dir)
        case['acoustic_dump'] = {}
        self.assertFalse(base_evidence(case)['dump_present'])

    def test_duplicate_utterance_id_rejected(self):
        directory = self.real_fixture()
        case = read_case(directory)
        case['utterances'].append(copy.deepcopy(case['utterances'][0]))
        write_json(directory / 'case.json', case)
        with self.assertRaisesRegex(ValueError, 'utt_id重复'):
            read_case(directory)

    def test_source_hash_and_path_guard(self):
        directory = self.real_fixture()
        read_case(directory, require_real=True)
        (directory / 'raw.log').write_text('changed', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'E_SOURCE_HASH'): read_case(directory)
        with self.assertRaisesRegex(ValueError, 'E_SOURCE_PATH'): source_path(directory, '../outside')

    def test_snapshot_records_audio_source(self):
        directory = self.real_fixture()
        snap = snapshot(directory, self.root / 'snap', ['raw.log'], ['raw.log'], 'g1', 'synthetic')
        self.assertEqual(snap['audio_fingerprints'][0], snap['sources'][0]['sha256'])
        self.assertEqual(read_case(self.root / 'snap')['group_id'], 'g1')

    def test_pathlib_and_mkdir_cannot_escape_sandbox(self):
        code = 'from pathlib import Path\nprint(Path(' + repr(str(ROOT / 'pyproject.toml')) + ').read_text())'
        self.assertFalse(run_code(code, self.case_dir).ok)
        self.assertFalse(run_code("import os\nos.mkdir('new-dir')", self.case_dir).ok)

    def test_token_mask_and_length(self):
        prompt = [{'role': 'user', 'content': 'question'}]
        answer = [{'role': 'assistant', 'content': 'answer'}]
        encoded = encode_example(ToyTokenizer(), prompt, answer, 1000)
        targets = [i for i in encoded['labels'] if i != -100]
        self.assertEqual(''.join(chr(i) for i in targets), 'answer<end>')
        text = ''.join(chr(i) for i in encoded['input_ids'])
        self.assertEqual(text.count('<bos>'), 1)
        self.assertIn('<empty_thought>answer', text)
        with self.assertRaisesRegex(ValueError, 'E_LENGTH'):
            encode_example(ToyTokenizer(), prompt, answer, 1)

    def test_bigram_really_updates_parameters(self):
        losses, generated, params = train_bigram(20)
        self.assertLess(losses[-1], losses[0])
        self.assertGreater(params, 1)
        self.assertEqual(len(generated), 11)


class GatewayTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name)
        self.endpoint = {'base_url': 'http://127.0.0.1:1/v1', 'model': 'mock',
                         'timeout_s': 1, 'max_tokens': 10, 'api_key_env': 'TEST_TEACHER_KEY'}

    def tearDown(self): self.tmp.cleanup()

    def test_config_refuses_credentials_and_unknown_hosts(self):
        for url in ['http://user:secret@127.0.0.1/v1', 'http://outside.invalid/v1']:
            write_json(self.path / 'c.json', {'allowed_hosts': ['127.0.0.1'], 'endpoints': {'teacher': self.endpoint | {'base_url': url}}})
            with self.assertRaises(ValueError): load_endpoint(self.path / 'c.json', 'teacher')

    def test_key_missing_fails_before_network(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(GatewayError, 'E_AUTH'): Client(self.endpoint, self.path, 'case')

    def test_response_and_logs_redact_key(self):
        with patch.dict(os.environ, {'TEST_TEACHER_KEY': 'sentinel-secret'}):
            client = Client(self.endpoint, self.path, 'case')
        reply = json.dumps({'choices': [{'finish_reason': 'stop', 'message': {'content': 'sentinel-secret'}}]})
        with patch.object(client.opener, 'open', return_value=io.BytesIO(reply.encode())) as mocked:
            self.assertEqual(client.chat([]), '[REDACTED]')
        self.assertEqual(mocked.call_count, 1)
        for path in self.path.iterdir():
            self.assertNotIn('sentinel-secret', path.read_text(encoding='utf-8'))

    def test_truncation_and_invalid_json_are_preserved(self):
        for index, reply in enumerate(['not json', '{"choices":[{"finish_reason":"length","message":{"content":"x"}}]}']):
            with patch.dict(os.environ, {'TEST_TEACHER_KEY': 'key'}):
                client = Client(self.endpoint, self.path / str(index), 'case')
            with patch.object(client.opener, 'open', return_value=io.BytesIO(reply.encode())) as mocked:
                with self.assertRaises(GatewayError): client.chat([])
            self.assertEqual(mocked.call_count, 1)  # 不盲重试
            self.assertEqual(len(list((self.path / str(index)).glob('*.response.txt'))), 1)

    def test_redirect_refused(self):
        with self.assertRaises(GatewayError): NoRedirect().redirect_request(None, None, 302, '', {}, 'http://other')


if __name__ == '__main__':
    unittest.main()
