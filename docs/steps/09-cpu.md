# 09 合并、量化和CPU部署

训练产物adapter是增量权重，需要原始基座。CPU服务准备流程是：本地基座+adapter → 合并模型 → GGUF → 量化文件 → llama-server → 同一个analyze命令。

先生成你自己的路径命令，不执行：

```bash
python -m ticket_sft.intranet deploy-plan \
  --model /models/gemma-4-12B-it \
  --adapter output/gemma_sft_v1/adapter \
  --merged /models/asr_gemma_v1 \
  --llama-dir /opt/llama.cpp
```

合并脚本用CPU FP32计算并safe_merge，保存BF16权重。FP32加载会比推理占更多内存；实际峰值取决于模型与库实现。只有在本地模型和依赖准备好后才执行输出命令。

转换器必须支持你的精确Gemma架构。先验证高精度GGUF，再Q8_0，再按实际质量考虑Q5_K_M等精度；转换成功不代表生成一致。保留原adapter和基座，不要只留下量化文件。

服务计划默认CPU `-ngl 0`、上下文4096、单并发、16线程、127.0.0.1:8080。16线程只是起点，需按实际CPU/NUMA测试。configs/intranet.json中student使用alias `asr-gemma`，并传关闭thinking参数；确认实际llama-server接受该参数，训练和推理模板需一致。

上下文必须容纳该轮完整输入加输出预算；若把训练长度从2K提高到4K，不能仍假定服务端4K上下文能容纳任意长输入再生成2K答案。先用同一tokenizer测量，并核对服务端是否截断历史。

## CentOS7与300GB内存

内存容量通常不再是小规模量化模型的主要限制，但内存带宽、CPU指令集、NUMA、上下文和输出长度决定速度。不能由300GB直接推断每天能处理多少条。CentOS7常见的glibc/libstdc++与新版预编译程序存在兼容问题，先在目标机检查版本并做CPU启动测试；在兼容基线构建或在受支持的隔离环境运行，不能盲拷贝新系统二进制。

部署验收顺序：模型加载 → 固定问答 → JSON动作 → 脚本运行 → 完整问题单 → 测试集语义对照 → 并发/长上下文吞吐。记录首token延迟、两轮总时长、CPU内存峰值和失败率。不要只用tokens/s代表业务吞吐。

Debug：GGUF不识别先查转换器/架构版本；CPU指令错误核对目标CPU编译选项；格式不一致查chat template和thinking；服务超时查排队和输出上限。
