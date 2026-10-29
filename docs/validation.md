# 本次验证记录

日期：2026-10-29。版本：0.2.0。

已完成：

- 原仓库 selftest 六个阶段通过，包含沙箱拦截、数据重放、debug和报告一致性。
- 新增23项标准库单元测试通过，覆盖网关响应/脱敏、动作与脚本校验、证据身份、候选修改、真实数据模式、跨集合音频泄漏、测试集隔离、人工评测覆盖、模板掩码与小矩阵梯度更新。
- 四个合成案例完整执行，得到4份候选、4份模拟审核记录、8条Gemma决策；train/val/test案例数为2/1/1。
- select导出测试输入、curate实际重放、evaluate核对输入快照的串联演练通过。人工技术评分保持未填写，未将演练当成语义验收。
- Python源码编译检查、shell脚本语法检查、Git空白检查通过。
- train默认仅打印计划，demo与基础实验不加载模型或调用网关。

测试中的tokenizer是最小替身，用来验证模板和mask处理逻辑，不是实际Gemma tokenizer。API响应使用mock，不代表内网服务兼容性已通过。合成数据只用于工程演练。

尚未执行：环境安装、模型下载、真实DeepSeek/GLM请求、V100模型加载或训练、checkpoint实际恢复、适配器合并、GGUF转换量化、CentOS7 CPU服务、真实诊断质量与日处理200条吞吐测试。

复核命令：

```bash
export PYTHONPATH="$PWD/src"
python -m ticket_sft.selftest
python -m unittest discover -s tests -v
python -m ticket_sft.intranet demo --out output/check_new_version
```

改变实现后只重复受影响的检查；改变模板、模型、数据、依赖或量化精度时，在目标机重做对应验收。
