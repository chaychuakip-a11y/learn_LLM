# 架构与代码导航

## 三条数据流

```text
训练准备：raw case → snapshot → teacher/gateway → pipeline → review → dataset
学习训练：lab实验 → training_text → train_gemma4 → adapter → evaluate
上线推理：CPU API → gateway → pipeline → sandbox/analysis → conclusion → 人审
```

旧工程的analysis/sandbox/rules/debug/build_dataset仍在；新入口复用这些实现而非拷入另一套课程工程。Gemma数据与推理使用protocol.py中相同的JSON文本动作。Qwen原生工具格式保留在dataset的swift产物里。

| 模块 | 输入 | 输出 | 失败边界 |
| --- | --- | --- | --- |
| snapshot/evidence | 结构化案例、声明附件 | 来源清单、音频指纹 | 路径越界、文件变化 |
| gateway | 内网端点配置、messages | 原始回复与请求日志 | 鉴权、超时、截断、重定向 |
| protocol | 回复文本 | run_analysis或final | 未声明动作、代码入口 |
| pipeline | 案例、可注入chat函数 | 逐步轨迹、候选 | 顺序错误、脚本失败、身份不符 |
| sandbox/analysis | 代码、只读案例 | JSON证据 | 写操作、目录外读取、超时 |
| review | 候选、人审信息 | cases/gold/meta | 未审核、源变化、重复金标 |
| dataset | 审核目录 | swift与gemma格式 | 跨组泄漏、审核后改动、空划分 |
| training_text | 原生模板、prompt/target | input_ids/labels | 长度与mask边界 |
| train_gemma4 | 本地权重和数据 | LoRA、checkpoint、指标 | 数值、显存、分布式 |
| evaluate | 冻结测试清单、预测、人审 | 分语言指标 | 缺失预测、输入快照不一致 |

## 两步训练序列

第一条：system + user(case) → assistant(JSON脚本动作)。

第二条：system + user(case) + assistant(JSON脚本动作) + user(实际工具结果) → assistant(JSON结论动作)。

第二条中的工具结果是上下文，标签为-100。第一条不含未来工具输出和最终结论。同case两条决策进入同一个split。每个模型输入保留原语种，不通过翻译替换LAB。

## 状态与可复现性

pipeline的最终状态只有failed或pending_review。批准来自显式人工review，规则通过不能自动提升状态。meta记录来源哈希、脚本/结论哈希、分组、审核人和备注。split_report是数据版本的固定清单。修改来源或目标后需要新版本与重新审核。

终端跟学入口只打印文档，lab实际执行小计算，demo实际执行本地脚本。只有teacher/analyze/train显式加--execute才会调用服务或训练。deploy-plan只打印命令，merge_adapter需另行显式执行。

## 未实现的适配边界

现有schema支持结构化资源、分支和声学路径观察，也可按来源读取文本附件。它不包含私有dump通用解析器、实时引擎重测/建图、完整WFST/FSA图分析、业务单状态写回、检索数据库。扩展这些功能时需要实际接口、样例和对应测试。
