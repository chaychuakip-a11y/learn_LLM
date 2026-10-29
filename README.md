# learn_LLM：从零做一个内网多语种问题单诊断模型

这是一个能跟着代码走完的工程：先用纯 Python 小实验理解 loss 和参数更新，再让分析脚本读取案例、dump 和资源证据，接入内网 DeepSeek / GLM 制作示范，审核后训练本地 Gemma，最后量化到 CPU 服务。每一步都有命令、输出、源码入口和 debug 方法。

这里的“从零”指从第一次接触训练开始学习。正式诊断模型从预训练权重做 LoRA；基础实验另外提供可执行的小矩阵训练。已有 Qwen/ms-swift 路线仍保留，见 [原路线](docs/legacy-qwen.md)。新增主线是 **4×V100 16GB / FP16 / ZeRO-3 / Gemma 候选**。

## 第一次打开，先跑这几条

在仓库根目录，Linux：

```bash
export PYTHONPATH="$PWD/src"
export PYTHONUTF8=1
python -m ticket_sft
python -m ticket_sft learn 0
python -m ticket_sft lab train-toy --steps 80
python -m ticket_sft.intranet demo --out output/first-run
python -m ticket_sft.intranet logs --run output/first-run
```

Windows PowerShell：

```powershell
$env:PYTHONPATH = 'src'
$env:PYTHONUTF8 = '1'
python -m ticket_sft
python -m ticket_sft lab train-toy --steps 80
python -m ticket_sft.intranet demo --out output/first-run
```

以上只需现有 Python 3.10+，不调用 API、不加载模型。再次运行 demo 时换一个输出目录。你会看到 4 个多语种合成案例依次执行，以及 8 条训练决策的生成。

```text
[#####---------------] 1/4 ar_media_003 pending_review 0.xxxs
[##########----------] 2/4 de_nav_002 pending_review 0.xxxs
[###############-----] 3/4 th_poi_001 pending_review 0.xxxs
[####################] 4/4 zh_contact_004 pending_review 0.xxxs
{"train": 2, "val": 1, "test": 1}
```

`pending_review` 是待人工审核，不是根因已确认。demo 读取仓库已有脚本与结论，模拟教师的两次回复；不伪装成真实模型生成。

## 跟学路线

每章可在终端阅读：`python -m ticket_sft learn N`。练习完成后用 `--done` 记录本地进度。原有八课概念材料也可通过 `python -m ticket_sft.basics` 阅读。

| 步骤 | 实际做什么 | 会留下什么 |
| --- | --- | --- |
| [00](docs/steps/00-start.md) | 跑通小矩阵训练和完整离线链路 | loss、候选、工具输出、训练JSONL |
| [01](docs/steps/01-evidence.md) | 对齐 LAB/REC，读分支和 dump | 终端对齐表、证据缺口 |
| [02](docs/steps/02-data.md) | 冻结真实案例和原始附件 | 来源哈希、原音频指纹、group_id |
| [03](docs/steps/03-teacher.md) | 请求 DeepSeek / GLM 并实际执行分析脚本 | 请求、响应、stdout、候选结论 |
| [04](docs/steps/04-review.md) | 人工审核与独立分组 | cases/gold、train/val/test |
| [05](docs/steps/05-training-math.md) | 理解 token、mask、loss、LoRA、累积 | 小实验和真实token预检 |
| [06](docs/steps/06-train.md) | 四卡20步冒烟，再正式训练 | 每卡日志、checkpoint、adapter |
| [07](docs/steps/07-debug.md) | 按日志逐层定位失败 | 最小复现、故障归类 |
| [08](docs/steps/08-eval.md) | 对照原模型、学生和教师 | 分语种评测、人工复核表 |
| [09](docs/steps/09-cpu.md) | 合并、GGUF、量化、CPU服务 | 可被同一诊断入口调用的学生 |
| [10](docs/steps/10-own-project.md) | 从20例试点走到自己的项目 | 数据版本、训练记录、回归结果 |

## 内网主流程

```text
问题单 + 原音频 + dump/资源/版本
  → snapshot：冻结来源
  → teacher：DeepSeek/GLM → 分析脚本 → 实际只读执行 → 结论
  → approve：人工对照证据，只选一个目标
  → build：分组隔离 + 重放 + swift/Gemma两种格式
  → tokens：实际tokenizer和答案mask预检
  → train：FP16 LoRA/ZeRO-3
  → evaluate：完整诊断 + 人工技术评分
  → merge / GGUF / CPU服务
  → analyze：新问题单，沿用同一动作协议和工具
```

详细接口见 [架构与代码导航](docs/architecture.md)，故障索引见 [DEBUG](docs/debug.md)，内网交付见 [部署与版本](docs/deployment.md)，已做和未做的检查见 [验证记录](docs/validation.md)。

## 工程目录

```text
src/ticket_sft/
  lab.py                  基础实验：实际梯度下降、loss、mask、LoRA
  analysis.py              LAB/REC、分支、词表、dump与已登记附件读取
  snapshot.py / evidence.py 原始来源冻结与核验
  gateway.py               内网API、鉴权、超时、原始请求响应
  protocol.py              教师、训练、推理共用的JSON动作协议
  pipeline.py              两步诊断状态机与每步日志
  review.py / dataset.py   人工审核入口、分组隔离、两种数据格式
  training_text.py         原生聊天模板和严格答案掩码
  train_gemma4.py          本地模型、四卡FP16 LoRA/ZeRO-3
  evaluate.py              测试集流程评测与人工质量评分
  merge_adapter.py         本地CPU合并适配器
  intranet.py              统一工作命令与终端日志视图
configs/                  端点、案例模板、ZeRO-3配置
data/example/             原有四条合成案例；不能充当真实训练数据
docs/steps/               每步命令、输入输出、代码说明、练习、debug
tests/                    标准库回归测试
output/                   本机运行产物，Git忽略
data/my/                  内网真实数据，Git忽略
```

## 验证与边界

```bash
python -m ticket_sft.selftest
python -m unittest discover -s tests -v
python -m ticket_sft.intranet doctor
```

本次只验证代码、离线演练和模拟接口，未安装环境、下载权重或启动 GPU 训练。内网网关、实际 Gemma tokenizer、四卡显存与稳定性、合并/GGUF、CentOS7/300GB CPU吞吐都须目标机验收。`requirements-gemma.txt` 是依赖清单，成功冒烟后再锁定实际版本；不要与旧 Qwen 的版本约束混装。

当前读取结构化 `case.json` 和已登记的 UTF-8 原始附件。私有二进制 dump 的字段解析、FSA/WFST 引擎工具适配仍需要结合内网工具实现。分析脚本的 Python 护栏用于学习和受控内网进程；生产执行生成代码时需额外采用低权限、无网络、只读挂载的进程/容器隔离。项目不自动修改资源、发布引擎或流转问题单。
