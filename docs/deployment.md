# 内网交付、依赖与版本

把本仓库代码、配置模板和文档带入内网；模型和依赖由你按内网条件准备。本次交付不执行安装或下载。运行代码始终引用本地模型目录，新网关只使用显式allowed_hosts中的地址。

## 两条训练路线

- 新Gemma路线：requirements-gemma.txt，FP16 LoRA + ZeRO-3，4×V100候选方案。Transformers需包含实际模型类，PyTorch/CUDA需能在Volta执行。先doctor，再实际小矩阵与20步冒烟。
- 原Qwen路线：requirements-train.txt、configs/sft_9b.env、scripts/train.sh。保留原行为与学习资料；其依赖、bf16、单卡内存假设不能直接套到V100。

不要把两个要求混进同一个未经验证的环境。成功运行后保存 `python -m pip freeze`、nvidia-smi、操作系统、实际基座revision/哈希以及训练命令。项目不提供假称验证过的固定版本锁。

## 配置与产物

configs/intranet.example.json可提交；configs/intranet.json、data/my、output、runs、模型权重和.env均被Git忽略。内部端点、业务数据、原始dump和密钥只保存在内网。审核候选的原始请求响应包含业务文本，按内部数据要求管理。

CPU服务先监听127.0.0.1，内部访问方式按现有部署体系配置。给模型生成脚本配独立低权限执行账户、无网络、只读输入和有资源限制的隔离进程。Python护栏与AST限制不能当成对敌对代码的完整操作系统安全边界。

## 资料依据

工程接口设计参考以下官方资料；内网需以实际成功环境为准：

- https://huggingface.co/google/gemma-4-12B-it
- https://huggingface.co/docs/transformers/main/model_doc/gemma4_unified
- https://huggingface.co/docs/transformers/main/deepspeed
- https://huggingface.co/google/gemma-4-12B-it/blob/main/chat_template.jinja
- https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md

尚未执行目标机训练、真实网关请求、权重合并、GGUF转换或CentOS7服务启动。离线测试和mock验证范围见tests与验证记录。
