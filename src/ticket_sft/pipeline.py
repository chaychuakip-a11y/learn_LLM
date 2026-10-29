"""真正的两阶段诊断：模型写脚本 → 实际执行 → 模型读结果写结论。

返回 pending_review 表示可供审核，不代表根因已经正确。所有外部调用由调用者
显式注入 chat，因而同一流程可以用离线示例、自建教师或 CPU 学生验证。
"""
import json
import time
from pathlib import Path

from ticket_sft.evidence import read_case, sha256
from ticket_sft.protocol import initial_messages, parse_action, tool_message
from ticket_sft.rules import check_conclusion, loads_stdout
from ticket_sft.runtime_log import RunLogger
from ticket_sft.sandbox import run_code


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def diagnose(case_dir, output, chat, policy, sandbox_timeout=8, log=None):
    case_dir, output = Path(case_dir).resolve(), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'result.json').exists():
        raise ValueError('E_OUTPUT_EXISTS: 请使用新的运行目录')
    case = read_case(case_dir)
    logger = log or RunLogger(output / 'events.jsonl', 'diagnosis')
    messages = initial_messages(case)
    # 保存原始快照，以便审核时检查源文件有没有变。
    write_json(output / 'case.snapshot.json', case)
    result = {'case_id': case['case_id'], 'case_dir': str(case_dir), 'policy': policy,
              'case_sha256': sha256(case_dir / 'case.json'),
              'status': 'failed', 'review_status': 'pending', 'semantic_verified': False}
    started = time.monotonic()
    try:
        logger.event('case_start', case_id=case['case_id'], policy=policy)
        for index in range(2):
            logger.event('model_start', stage=index + 1, input_messages=len(messages))
            raw = chat(messages)
            (output / f'model_{index + 1}.txt').write_text(raw, encoding='utf-8')
            action = parse_action(raw)
            messages.append({'role': 'assistant', 'content': raw})
            if index == 0:
                if action['action'] != 'run_analysis':
                    raise ValueError('E_SEQUENCE: 第一步必须执行分析脚本')
                code = action['code']
                (output / 'analysis.py').write_text(code, encoding='utf-8')
                tool_started = time.monotonic()
                logger.event('tool_start', tool='run_analysis', code_chars=len(code))
                executed = run_code(code, case_dir, timeout=sandbox_timeout)
                tool_result = vars(executed)
                write_json(output / 'tool_result.json', tool_result)
                (output / 'stdout.json').write_text(executed.stdout, encoding='utf-8')
                (output / 'stderr.log').write_text(executed.stderr, encoding='utf-8')
                logger.event('tool_done', ok=executed.ok, exit_code=executed.exit_code,
                             latency_s=round(time.monotonic() - tool_started, 3),
                             stdout_chars=len(executed.stdout), stderr_chars=len(executed.stderr))
                if not executed.ok:
                    raise ValueError('E_TOOL_FAILED: 脚本执行失败，请检查 stderr.log')
                evidence = loads_stdout(executed.stdout)
                if evidence.get('case_id') != case['case_id']:
                    raise ValueError('E_IDENTITY: stdout.case_id 与当前问题单不一致')
                messages.append(tool_message(tool_result))
            else:
                if action['action'] != 'final':
                    raise ValueError('E_SEQUENCE: 第二步需要结论；最多一次脚本执行')
                conclusion = action['conclusion']
                (output / 'conclusion.md').write_text(conclusion, encoding='utf-8')
                findings = [vars(f) for f in check_conclusion(evidence, conclusion)]
                result['rule_findings'] = findings
                if any(f['level'] == 'error' for f in findings):
                    raise ValueError('E_RULES: 结论未通过有限规则检查；查看 result.json')
        result['status'] = 'pending_review'
    except Exception as exc:
        result['error_type'] = type(exc).__name__
        result['error'] = str(exc)
        logger.event('case_failed', level='ERROR', error_type=type(exc).__name__, message=str(exc))
    finally:
        result['duration_s'] = round(time.monotonic() - started, 3)
        write_json(output / 'transcript.json', messages)
        # 审核时重算这些哈希，发现手改文件后要求重新生成候选。
        result['artifact_hashes'] = {p.name: sha256(p) for p in output.iterdir()
                                     if p.is_file() and p.name in
                                     {'analysis.py', 'conclusion.md', 'stdout.json', 'tool_result.json', 'transcript.json'}}
        write_json(output / 'result.json', result)
        logger.event('case_done', status=result['status'], duration_s=result['duration_s'])
    return result
