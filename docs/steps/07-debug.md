# 07 每步如何debug

先看阶段边界：输入快照 → API → 动作解析 → 脚本执行 → 结论 → 审核 → 分词 → GPU → 推理。某一阶段失败，不跳到后面用调参掩盖。

```bash
python -m ticket_sft.intranet logs --run output/teachers/ds_v1 --tail 20
python -m ticket_sft.intranet logs --run output/gemma_sft_v1 --follow
python -m ticket_sft.debug --cases data/my/reviewed_v1/cases --gold data/my/reviewed_v1/gold
```

debug保留原仓库终端对齐、分支和token占用条。新日志视图显示elapsed/event/level/step/loss/VRAM。同一API案例的gateway与diagnosis共享trace_id，request_id对应请求地址与正文的哈希。原ASR测试run_id与这些诊断trace不是一个概念。

## 对一个失败最小复现

1. 打开result.json，找error和失败阶段。
2. 对照case.snapshot.json，确认当时输入与当前源文件一致。
3. API错误看api/*.request.json和response.txt；脚本错误看analysis.py与stderr.log。
4. 只重放这个脚本：

```bash
python -m ticket_sft.sandbox --case data/my/snapshots/ticket_001 \
  --code output/teachers/ds_v1/ticket_001/analysis.py
```

5. 需要逐行单步时，用Python自带pdb调试**编排代码**：

```bash
python -m pdb -m ticket_sft.intranet demo --out output/pdb_run_01
```

pdb中 `b src/ticket_sft/pipeline.py:行号` 设置断点，`n`下一行、`s`进入、`p messages`看变量、`c`继续。先查看源码确定实际行号。生产API key不要打印；模型权重也无需为调试数据阶段加载。

## 训练问题常用顺序

OOM先检查是否ZeRO-3在模型加载前生效，再看最长样本、激活、保存时是否聚合。loss=NaN先看首个异常step和FP16数值，不是简单把日志过滤打开。只有rank0前进时对齐各卡最后事件与console。重载像原模型时查adapter、基座版本和固定对照样本。

详见docs/debug.md的故障表。练习：在演练副本制造一次JSON动作错误、一次脚本异常、一次长度超限，分别找到对应日志。验收是能保存最小复现，不是能背错误名。
