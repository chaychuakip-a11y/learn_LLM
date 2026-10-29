"""标准库内网 Chat Completions 客户端；不读取公开端点默认值。"""
import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from ticket_sft.runtime_log import RunLogger, redact


class GatewayError(RuntimeError):
    pass


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise GatewayError('E_REDIRECT: 网关重定向被拒绝，请配置最终地址')


def load_endpoint(path, name):
    config = json.loads(Path(path).read_text(encoding='utf-8'))
    endpoint = dict(config['endpoints'][name])
    url = urllib.parse.urlsplit(endpoint['base_url'])
    if url.scheme not in {'http', 'https'} or not url.hostname or url.username or url.password or url.query or url.fragment:
        raise ValueError('E_ENDPOINT: 需要无凭据和查询参数的 HTTP(S) base_url')
    if url.hostname not in config.get('allowed_hosts', []):
        raise ValueError('E_ENDPOINT: 主机不在配置 allowed_hosts 中')
    if not isinstance(endpoint.get('model'), str) or not endpoint['model'].strip():
        raise ValueError('E_ENDPOINT: 必须填写网关实际模型 ID')
    if any('REPLACE' in str(endpoint.get(k, '')) for k in ['base_url', 'model']):
        raise ValueError('E_CONFIG: 请先替换 REPLACE 占位配置')
    if endpoint.get('protocol', 'json') != 'json':
        raise ValueError('E_PROTOCOL: 本入口使用 JSON 动作协议')
    for key, default in [('timeout_s', 180), ('max_tokens', 2048)]:
        endpoint[key] = float(endpoint.get(key, default)) if key == 'timeout_s' else int(endpoint.get(key, default))
        if endpoint[key] <= 0:
            raise ValueError('E_CONFIG: timeout_s/max_tokens 必须大于零')
    return endpoint


class Client:
    def __init__(self, endpoint, directory, case_id):
        self.endpoint = endpoint
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        key_env = endpoint.get('api_key_env')
        self.key = os.environ.get(key_env, '') if key_env else ''
        if key_env and not self.key:
            raise GatewayError(f'E_AUTH: 未设置环境变量 {key_env}')
        self.log = RunLogger(self.directory / 'events.jsonl', 'gateway', secrets=(self.key,))
        self.case_id = case_id
        self.count = 0
        # 内网调用不沿用机器上的公网代理；不自动跟随重定向。
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())

    def chat(self, messages):
        self.count += 1
        payload = {'model': self.endpoint['model'], 'messages': messages,
                   'temperature': 0, 'max_tokens': self.endpoint['max_tokens']}
        extra = self.endpoint.get('extra_body', {})
        if not isinstance(extra, dict) or set(extra) & set(payload) or 'stream' in extra:
            raise GatewayError('E_CONFIG: extra_body 不得覆盖 messages/model/temperature/max_tokens/stream')
        payload.update(extra)
        url = self.endpoint['base_url'].rstrip('/') + '/chat/completions'
        data = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        request_id = hashlib.sha256(url.encode() + data).hexdigest()[:20]
        stem = f'{self.count:02d}_{request_id}'
        self._save(stem + '.request.json', payload)
        headers = {'Content-Type': 'application/json'}
        if self.key:
            headers['Authorization'] = 'Bearer ' + self.key
        started = time.monotonic()
        self.log.event('request_start', case_id=self.case_id, request_id=request_id,
                       model=self.endpoint['model'], request_bytes=len(data))
        try:
            req = urllib.request.Request(url, data=data, headers=headers)
            with self.opener.open(req, timeout=self.endpoint['timeout_s']) as response:
                raw = response.read(8 * 1024 * 1024 + 1)
            if len(raw) > 8 * 1024 * 1024:
                raise GatewayError('E_RESPONSE_SIZE: 响应超过 8 MiB')
            text = raw.decode('utf-8')
            self._save(stem + '.response.txt', text, raw=True)
            reply = json.loads(text)
            choice = reply['choices'][0]
            if choice.get('finish_reason') != 'stop':
                raise GatewayError('E_FINISH: 未完整结束，finish_reason=' + str(choice.get('finish_reason')))
            content = choice['message'].get('content')
            if not isinstance(content, str) or not content.strip():
                raise GatewayError('E_CONTENT: 没有文本答案；检查 reasoning/content 网关格式')
            self.log.event('request_done', case_id=self.case_id, request_id=request_id,
                           latency_s=round(time.monotonic() - started, 3), usage=reply.get('usage'))
            return redact(content, (self.key,))
        except Exception as exc:
            if isinstance(exc, urllib.error.HTTPError):
                body = exc.read(65536).decode('utf-8', errors='replace')
                self._save(stem + '.http_error.txt', body, raw=True)
            self.log.event('request_failed', level='ERROR', request_id=request_id,
                           error_type=type(exc).__name__, message=str(exc),
                           latency_s=round(time.monotonic() - started, 3))
            raise GatewayError(redact(str(exc), (self.key,))) from None

    def _save(self, name, value, raw=False):
        cleaned = redact(value, (self.key,))
        (self.directory / name).write_text(cleaned if raw else json.dumps(cleaned, ensure_ascii=False, indent=2), encoding='utf-8')
