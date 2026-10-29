# 03 用内网 DeepSeek 和 GLM 制作示范

你已经有兼容 Chat Completions 的内网接口。只填写自己的地址、实际model ID、API key环境变量名。这里不需要logits；通过审核的输出和工具轨迹就能做监督蒸馏。

```bash
cp configs/intranet.example.json configs/intranet.json
```

编辑 allowed_hosts 和 endpoints。key存在环境变量，不写入JSON或Git。内网是否无key，以网关实际要求为准；无鉴权时将api_key_env设为null。base_url填到 `/v1`，不要重复填写 `/chat/completions`。extra_body可填写该网关支持的思考参数，不能覆盖messages/model/temperature/max_tokens。

先检查计划：

```bash
python -m ticket_sft.intranet teacher --config configs/intranet.json \
  --endpoint deepseek --case data/my/snapshots/ticket_001 --out output/teachers/ds_v1
```

核对后加 `--execute` 实际调用。同样用 `--endpoint glm --out output/teachers/glm_v1` 获取另一份候选。批量使用 `--cases data/my/snapshots --limit 20`，顺序执行，便于定位失败。

## 实际发生了什么

1. initial_messages送出规程、工具接口和案例。
2. 教师返回JSON：`{"action":"run_analysis","code":"..."}`。
3. validate_code检查代码接口，run_code在子进程实际运行。
4. 将stdout/stderr等真实结果作为下一轮上下文。
5. 教师返回JSON：`{"action":"final","conclusion":"..."}`。
6. 保存候选，等待人工核对。

JSON动作协议是项目自己的文本约定，不依赖网关原生function-calling。Gemma训练与CPU推理也使用同一协议。这个版本只允许一次脚本调用和一次结论；需要多轮深查时先扩展协议、数据和测试，再训练。

阅读 `gateway.Client.chat` 与 `pipeline.diagnose`，断点可打在parse_action和run_code前后。请求目录api保存 `.request.json`、`.response.txt` 和events。认证头不写日志；业务文本会保存在本地运行目录。

## Debug与验收

- 401/403：核对key环境变量、服务权限，不重跑训练。
- 404：核对base_url和实际模型ID。
- content为空：看原始响应是不是只有reasoning。
- finish_reason=length：答案被截断，调整输出长度或输入规模后新开批次。
- 脚本失败：查stderr，不让第二轮把失败解释成“资源不存在”。

程序不盲重试；一次超时可能是服务端已计算但客户端没收到。保留批次后只重做已查明需要重试的案例。两位教师一致仍不等于根因真值。

验收：先完成一条案例，确认教师看到的输入、实际脚本输出、候选结论三者对应，再处理20条。
