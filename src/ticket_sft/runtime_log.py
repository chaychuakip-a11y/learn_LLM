"""结构化日志：关联一次运行和案例，不记录认证头。"""
import datetime
import json
import time
import uuid
from pathlib import Path

SECRET_KEYS = {'authorization', 'api_key', 'password', 'access_token', 'refresh_token', 'secret'}

def redact(value, secrets=()):
    if isinstance(value, dict):
        return {k: '[REDACTED]' if k.lower() in SECRET_KEYS else redact(v, secrets)
                for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(v, secrets) for v in value]
    if isinstance(value, str):
        for secret in secrets:
            if secret:
                value = value.replace(secret, '[REDACTED]')
    return value

class RunLogger:
    def __init__(self, path, component, secrets=(), run_id=None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.component = component
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.secrets = secrets
        self.started = time.monotonic()
    def event(self, event, level='INFO', **fields):
        record = {'timestamp_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  'elapsed_s': round(time.monotonic()-self.started, 4),
                  'trace_id': self.run_id, 'run_id': self.run_id, 'component': self.component, 'level': level,
                  'event': event, **fields}
        with self.path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(redact(record, self.secrets), ensure_ascii=False, default=str)+'\n')
        return record
