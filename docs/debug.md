# Debug索引

先定位“最后一个成功事件”，再检查下一步输入。保留失败目录，复现时用新输出路径。

| 现象/错误 | 首先看哪里 | 下一步 |
| --- | --- | --- |
| 找不到ticket_sft | PYTHONPATH与当前目录 | 按README从仓库根目录运行 |
| E_OUTPUT_EXISTS | 输出路径 | 新建版本或训练显式resume |
| E_SOURCE_PATH | sources的相对路径 | 核对附件在案例目录内 |
| E_SOURCE_HASH | 来源文件和case.snapshot | 重新冻结并重新审核 |
| E_REAL/E_PROVENANCE | data_origin/group/sources/audio | 补真实来源，不改假标记绕过 |
| E_AUTH / 401 / 403 | API环境变量、http_error | 核对网关鉴权与权限 |
| 404 | request里的base_url/model | 确认/v1和真实模型ID |
| E_REDIRECT | 端点配置 | 填最终内网地址 |
| timeout | request_start与服务端记录 | 先确认是否已完成，再决定重试 |
| E_CONTENT | 原始response | 区分reasoning与content |
| E_FINISH | finish_reason | 截断/拒绝不能当完整答案 |
| JSONDecodeError | model_N.txt或response.txt | 修正协议/长度，不补造答案 |
| E_ACTION/E_SEQUENCE | transcript.json | 先脚本后结论，共两轮 |
| E_SCRIPT_* | analysis.py | 使用声明的只读analysis接口 |
| E_TOOL_FAILED | stderr.log/tool_result | 重放脚本，检查退出码 |
| stdout非JSON | stdout.json | 只输出一个JSON对象 |
| E_IDENTITY | stdout.case_id | 核对当前案例，禁止串单 |
| E_RULES | result.rule_findings | 回读证据和五个标题 |
| E_CANDIDATE_CHANGED | 产物hash | 重新采集或创建新审核版本 |
| E_REVIEW_STALE | meta里的三类hash | 重新审核改动内容 |
| E_DUPLICATE | cases/gold已有案例 | 两位教师只能选一个金标 |
| E_LEAKAGE | group_splits/audio_splits | 合并同源组，保持历史划分 |
| E_EMPTY_SPLIT | split_report/元数据 | 补独立案例，不随意搬测试组 |
| E_TEMPLATE/E_TOKEN_BOUNDARY | tokenizer和模板预览 | 适配实际原生模板 |
| E_LENGTH | total_tokens/target_tokens | 提炼阶段上下文或验证更长长度 |
| CUDA no kernel image | console与CUDA能力 | 检查PyTorch包含Volta算子 |
| BF16/FA2不支持 | 训练precision/attention | V100主线使用FP16/eager |
| 初始OOM | ZeRO初始化与峰值显存 | 确认分片、缩短样本、检查激活 |
| 保存OOM | checkpoint保存事件 | 检查聚合峰值及ZeRO配置 |
| NaN/Inf | 首个异常loss/grad_norm | 查数据、FP16数值与算子 |
| 某rank不前进 | 各rank最后事件/console | 检查异常与通信，别只看rank0 |
| checkpoint不能恢复 | 基座/依赖/数据版本 | 用冒烟先验证保存恢复 |
| loss低效果差 | 独立测试与人审 | 查标签、覆盖、过拟合与泄漏 |
| adapter像没加载 | 模型和adapter路径 | 固定样本比较前后生成 |
| E_EVAL_SNAPSHOT | 测试输入sha256 | 使用冻结reviewed案例副本 |
| 格式通过但乱归因 | review_sheet人工栏 | 规则检查不能替代技术审核 |
| GGUF不识别 | converter版本/架构 | 先高精度转换加载，再量化 |
| CPU启动崩溃 | glibc/libstdc++/CPU指令 | 使用目标兼容构建 |
| CPU耗时高 | 全请求latency/上下文 | 先单并发，测线程/NUMA再调整 |

日志不记录认证头。网关会对配置的API key做脱敏；业务正文仍是业务数据，不会自动脱敏为公开材料。原始文件、stdout和模型结论的身份可核验，不代表内容已经正确。
