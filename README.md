# 边学边训练

这个目录训练一个内网分析员：读多语种车厂问题单，写只读 Python，沙箱执行，再根据标准输出写结论。基座是 Qwen3.5-9B，方式是 bf16 LoRA。底模不动，只训练旁边一小块权重。

示例数据是假的，用来把流程跑通。中文联系人、泰语 POI、德语导航、阿拉伯语音乐，各代表一种判法。真实问题单按同样的目录换进来。

第 0 课不需要 GPU，在当前机器就能做。装环境和训练放到内网 Linux。

模型本身的八课在 `lessons/`。一次读一课：

```text
python -m ticket_sft.basics
python -m ticket_sft.basics 1
```

每课都按新手写：先解释词，再用「打电话给张三」这条样本走一遍，最后有一步要你亲手做。一次只读一课。前四课放在看样本之前，第 5 到第 8 课留到第一次冒烟训练前后。

## 第 0 课：看懂一条样本

Linux：

```bash
bash scripts/learn.sh
```

Windows PowerShell：

```powershell
$env:PYTHONPATH = "src"
$env:PYTHONUTF8 = "1"
python -m ticket_sft.selftest
python -m ticket_sft.build_dataset
python -m ticket_sft.inspect --id zh_contact_004
python -m ticket_sft.report
```

然后打开 `output/debug/report.html`。

先看 `zh_contact_004`，再看 `th_poi_001`。两条都是最终 REC 错了，原因不同：

- 中文联系人按槽准。张三不在词典里，槽是空的。
- 泰语 POI 里 `เซ็นทรัล` 在词典里，但 WFST 路径不可达，而且没有进入候选。词在词典里不等于路径可达。
- 德语导航的正确文本在 fsa 候选里，但输给了 patch。dump 有 state 数字，但没有做声学模型映射，音素层只能待核实。
- 阿拉伯语音乐是云端结果，没有分支分数，也没有声学 dump，只能写成缺词。

`inspect` 里标着 `[要学]` 的是脚本和结论，训练时算 loss。`[上下文]` 是系统提示、问题单和脚本输出，不算 loss。脚本输出是证据，不能让模型再背一遍。

改完脚本想立刻看结果，不必训练：

```bash
python -m ticket_sft.sandbox --case data/example/cases/de_nav_002 --code data/example/gold/de_nav_002.py
```

沙箱会拦住写文件、联网和子进程。它是训练和调试用的护栏，不是对外的安全边界。

## 你要准备的数据

一条真实样本是一个目录加三个文件：

```text
data/my/cases/<单号>/case.json
data/my/gold/<单号>.py
data/my/gold/<单号>.md
data/my/gold/<单号>.meta.json
```

`case.json` 放车厂、语种、句准或槽准、分好词的标注和 REC、分支分数、词典、WFST、声学 dump。没有的字段写 `null` 或空数组，不要编。文本在进 JSON 之前先按你们引擎的分词和归一化处理好。示例为了能看见过程，用了空格分词和整句精确匹配。

`.py` 是分析员当时会写的脚本，只能 `import analysis` 并打印 JSON。`.md` 用这五个标题：口径、脚本在查什么、结论、待核实、正式门禁。结论里的每句话都要能在脚本输出里找到。`.meta.json` 写 `{"split": "train"}` 或 `{"split": "val"}`。

验证集按车厂和语种留出，不要只留一条同语种。示例里的 1 条验证只证明流程没断。

换到自己的数据时，改 `configs/sft_9b.env` 里这三行：

```text
DATA_CASES=data/my/cases
DATA_GOLD=data/my/gold
DATA_OUT=data/processed
```

然后重新 `python -m ticket_sft.build_dataset --cases data/my/cases --gold data/my/gold`。构建会重放脚本。脚本报错、结论越过证据、或者粗估 token 超过 `MAX_LENGTH`，都不会写出训练集。

一条样本先控制在大约十句以内。当前示例最长粗估约 2900 token，配置里的 `MAX_LENGTH` 是 4096。更长的单拆开，或者提示词里只放摘要，原始日志留在 `CASE_DIR` 给脚本读。

## 第 1 课：内网 Linux 上装环境

用 Python 3.10 到 3.12，一块 NVIDIA GPU。9B 的 bf16 LoRA 大约 22GB，24GB 卡是起点。16GB 卡用 4B。Qwen3.5 不默认做 4bit 训练，这个结构的量化误差比常见模型大。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U -r requirements-train.txt
pip install -U "flash-linear-attention>=0.4.2" --no-build-isolation
pip install -U git+https://github.com/Dao-AILab/causal-conv1d --no-build-isolation
```

`flash-attn` 可以后装。没装时用默认注意力，先求跑通。`preflight` 会告诉你 `fla` 和 `causal_conv1d` 有没有导入成功。没有它们时，Qwen3.5 的线性注意力会很慢，或者直接报错。

模型放到本地，训练机不必访问外网：

```bash
# 有 ModelScope 镜像时
modelscope download --model Qwen/Qwen3.5-9B --local_dir /data/models/Qwen3.5-9B
```

然后把 `configs/sft_9b.env` 的 `MODEL` 改成这个目录。算力低于 8.0 的卡把 `TORCH_DTYPE` 改成 `float16`。

## 第 2 课：先看命令，再冒烟

```bash
bash scripts/train.sh --dry-run
```

这一步不占 GPU。它会检查数据，并把真正的 `swift sft` 命令写到 `output/debug/train_command.sh`。打开这个文件，对着 README 下面的词表看每个参数。

确认数据通过之后：

```bash
bash scripts/train.sh --smoke
```

这是 20 步，用来确认 loss 会打出来、显存没爆、检查点能保存。示例训练集只有 3 条，这 20 步会把它们反复看多遍，不能当成正式效果。另开一个终端：

```bash
bash scripts/board.sh
```

或者训练停了以后：

```bash
python -m ticket_sft.plot_log --output output/sft_9b
```

图在 `output/debug/loss.svg`，蓝线是训练 loss，橙线是验证 loss。前 20 步跳动是正常的。冒烟的橙线几乎没有意义，验证集太小。

显存爆了按这个顺序改，改完再冒烟：

1. 确认 `GRADIENT_CHECKPOINTING` 仍是 `true`。
2. 把 `MAX_LENGTH` 降到 2048，并让 `check_data` 重新通过。通不过就缩短问题单，不要先砍结论。
3. 换成 4B：`bash scripts/train.sh --4b --smoke`。

## 第 3 课：正式训练

```bash
bash scripts/train.sh
```

默认 3 个 epoch，学习率 `1e-4`，LoRA rank 8。这是 Qwen3.5 官方 LoRA 示例的量级。loss 一直振荡，把 `LEARNING_RATE` 改成 `5e-5`。日志在 `output/sft_9b/train.log`。

训练中可以看的东西：

| 现象 | 用什么看 |
| --- | --- |
| 样本被截断或结论越过证据 | `python -m ticket_sft.check_data` |
| 不知道模型在学哪一段 | `python -m ticket_sft.inspect --id <单号>` |
| 脚本改完输出变了没有 | `python -m ticket_sft.sandbox` |
| 训练起不来 | `bash scripts/train.sh --dry-run` |
| loss 曲线 | TensorBoard，或 `python -m ticket_sft.plot_log` |
| 训练后模型会不会乱下声学结论 | `python -m ticket_sft.agent_loop --server ...` |

正式训练前把 `LOGGING_STEPS` 从 1 改成 5 或 10。数据变大之后，再把 `DATASET_NUM_PROC` 和 `DATALOADER_NUM_WORKERS` 调高。第一轮保持 1 和 0，报错更好读。`swift` 若说不认识 `group_by_length`，把 `output/debug/train_command.sh` 里那一行删掉，以 `swift sft -h` 为准。

## 训练之后怎么分析一张新单

先把适配器服务起来。参数以你机器上的 `swift deploy -h` 为准，常见写法是：

```bash
swift deploy \
  --adapters output/sft_9b/v0-xxxx/checkpoint-xxx \
  --infer_backend transformers \
  --port 8000
```

然后：

```bash
python -m ticket_sft.agent_loop \
  --case data/my/cases/<单号> \
  --server http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen3.5-9B
```

循环最多两轮：模型写脚本，沙箱执行，标准输出交回去，模型写结论。结束时用和训练集同一套规程检查。没有 dump 却写「已完成音素级验证」，这里会报出来。

服务还没起来时，用金标脚本把同一条链路先走一遍：

```bash
python -m ticket_sft.agent_loop \
  --case data/example/cases/th_poi_001 \
  --script data/example/gold/th_poi_001.py
```

## 这几个词

- **LoRA**：旁边加的小权重。`output/sft_9b/.../checkpoint-*` 里是它，不是一整个新模型。
- **step**：一次参数更新。每次先看 1 条，攒满 8 条才更新。样本少于 8 条时，一个 epoch 结束也会把没攒满的那次更新掉。
- **epoch**：训练集过完一遍。3 个 epoch 就是过三遍。样本只有几十条时，三遍容易背结论，优先增加不同车厂和语种的单，而不是把 epoch 加到 20。
- **loss**：预测下一个 token 的误差。只统计脚本和结论。它下降说明更像你的写法，不说明门禁判断已经对。
- **eval loss**：留出的验证集上的误差。验证集必须按车厂、语种留出，否则这条线只是在看训练样本。
- **适配器**：上面的 LoRA 文件。推理时基座加适配器一起用。

## 目录

```text
lessons/                   八课基础知识
prompts/system.md          分析规程，改完要重新 build_dataset
configs/sft_9b.env         9B 训练旋钮
configs/sft_4b.env         显存不够时用
data/example/              四条教学样本
data/my/                   你自己的案例和金标
src/ticket_sft/analysis.py 脚本可以调用的只读函数
src/ticket_sft/sandbox.py  执行脚本
src/ticket_sft/rules.py    结论能不能越过证据
```

规程原文的交付、邮件和上架不在模型的输出里。模型只负责把一张单分析到「结论 / 待核实 / 还不能当正式门禁」。正式门禁仍要固定基线上的完整平台结果。
