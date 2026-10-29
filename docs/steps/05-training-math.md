# 05 从小实验理解真正的训练代码

先分别运行：

```bash
python -m ticket_sft lab loss
python -m ticket_sft lab mask
python -m ticket_sft lab lora
python -m ticket_sft lab batch
python -m ticket_sft lab train-toy --steps 80
```

token是分词器的单位，不等于字符或单词。模型为下一个token给出logits，softmax得到概率。正确token概率越低，交叉熵 `-log(p)` 越大。loss是对训练标签的拟合误差，不是资源定位准确率。

mask里的 `-100` 是训练实现约定：这些标签位置忽略loss。系统提示、问题单、工具结果作为上下文；脚本动作、结论动作和原生结束符参与学习。一个teacher-forcing样本同时提供完整目标，模型内部通过因果注意力只能利用当前位置之前的信息。

LoRA冻结原W，在旁边训练A和B，使用 `W + alpha/rank × B×A`。这降低可训练参数和优化器存储，但基座、激活仍占内存。四张16GB卡不能自动当成一张64GB卡；ZeRO-3把参数等状态分片，激活仍需要每卡空间。

每卡batch=1、4卡、累积8，名义有效batch=32条决策。处理32条后更新一次参数；累计前向/反向次数和优化器step不是一个数。epoch表示训练集过一遍。数据少时反复很多epoch很容易记住答案。

## 看真实tokenizer的结果

模型目录已在内网准备好后：

```bash
python -m ticket_sft.intranet tokens --model /models/gemma-4-12B-it \
  --data data/my/dataset_v1/gemma/train.jsonl --max-length 2048 \
  --out output/token_check_v1.json
```

只加载本地processor/tokenizer，不加载权重。再对valid/test分别检查。看 total_tokens、target_tokens、target_preview。目标应从JSON动作开始，不能从问题单或dump开始。

阅读 `training_text.encode_example`：用原生推理前缀、原始目标和原生结束符构造样本；显式关闭thinking；检查模板和分词边界；超过max_length报错，不截断结论。某些Gemma模板的非思考生成前缀带空thought块，训练时必须与推理一致。

练习：把max-length设得极小，确认E_LENGTH；再检查一条目标结束符。2K不够时先按排查阶段减少上下文，或在实际显存验证后增大，不能删掉关键反证。

验收：能解释为何loss正常但诊断错误、为什么工具输出不算loss，以及为何实际token长度必须由目标模型的tokenizer检查。
