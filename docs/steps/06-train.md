# 06 在四张V100上试训，再正式训练

这一步留给内网已配置的Linux训练机。项目本身不自动安装依赖或下载模型。Gemma主线需要requirements-gemma.txt中的组件，并且Transformers实际包含Gemma4UnifiedForConditionalGeneration。不要复用旧Qwen的transformers<5.3约束。

先只读检查：

```bash
python -m ticket_sft.intranet doctor --hardware
python -m ticket_sft.intranet train --model /models/gemma-4-12B-it \
  --data data/my/dataset_v1/gemma --out output/gemma_smoke_v1 --smoke
```

第二条默认仅打印命令。核对训练数据、token检查、模型目录、四卡后，加 `--execute` 启动20步冒烟。模拟数据只在同时指定 `--smoke --allow-demo-smoke` 时允许进入训练，正式任务使用真实审核集。

## 脚本内部发生什么

1. torchrun启动4个进程，每个对应一张卡。
2. 每卡先实际执行FP16小矩阵运算，发现Volta算子不兼容。
3. 在加载模型前创建TrainingArguments，让ZeRO-3参与初始化。
4. 读取本地权重，不使用推理用的device_map=auto。
5. 严格编码输入，答案外标签设为-100。
6. 给语言层线性模块加rank16 LoRA，冻结大部分参数。
7. 每步记录loss、grad_norm、学习率及显存；非有限数值停止。
8. 保存中间checkpoint、最终adapter、processor和run_manifest。

默认FP16、eager attention、梯度检查点、每卡batch1、累积8、学习率5e-5、1epoch、2K上下文。V100无原生BF16；不默认使用要求更新GPU架构的FA2。12B是待验证候选，不承诺四卡16GB一定装得下所有长输入；先用真实样本观察峰值和稳定性。

```bash
python -m ticket_sft.intranet logs --run output/gemma_smoke_v1 --follow
```

看每卡 `debug/rank_N.jsonl`，主控制台同步tee到console.log。日志包括CUDA能力、实际模块名、序列长度、目标token数、显存allocated/reserved/peak、训练指标、保存事件。导入依赖阶段失败可能尚无JSONL，此时先读console.log。

## 恢复与正式训练

```bash
python -m ticket_sft.intranet train --model /models/gemma-4-12B-it \
  --data data/my/dataset_v1/gemma --out output/gemma_smoke_v1 \
  --smoke --resume output/gemma_smoke_v1/checkpoint-10 --execute

python -m ticket_sft.intranet train --model /models/gemma-4-12B-it \
  --data data/my/dataset_v1/gemma --out output/gemma_sft_v1 --execute
```

恢复checkpoint是恢复训练状态；加载adapter继续推理是另一件事。保持数据、基座、依赖版本一致，先验证一次保存恢复，再长训。checkpoint会包含分片状态，不能只复制一个rank的文件。

验收：真实20步有有限loss，所有卡前进，保存可恢复，adapter重载后能用同一格式生成。loss下降不是最终验收，下一步要对独立测试案例生成诊断。
