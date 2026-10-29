# 08 证明学生有用，比较原模型、教师和量化模型

测试集按split_report.json冻结。服务端用哪个模型，配置不同endpoint别名；所有模型看相同快照，使用相同工具。测试答案和修复后材料不得通过案例检索库泄漏给模型。

将**已审核目录中属于test的案例**导出到测试入口，保持case.json与附件字节不变，不把gold脚本或结论交给模型：

```bash
python -m ticket_sft.intranet select --dataset data/my/reviewed_v1 \
  --split test --out data/my/test_cases_v1

python -m ticket_sft.intranet analyze --config configs/intranet.json \
  --endpoint student --cases data/my/test_cases_v1 \
  --out output/eval_student_v1 --execute

python -m ticket_sft.evaluate --dataset data/my/dataset_v1 \
  --run output/eval_student_v1 --out output/metrics_student_v1
```

evaluate会对照所有预期测试case。缺失预测算流程未通过，输入快照哈希不一致会报错，不能只统计成功输出。结果按语种汇总，duration包括两次模型调用与工具执行。小样本p95只能作观察。

自动检查只能判断流程与有限规则。`review_sheet.json` 中资源定位、无依据根因断言、缺证识别默认null。请专家填写true/false、reviewer和note，不能用教师自评直接当真值。

```bash
python -m ticket_sft.evaluate --dataset data/my/dataset_v1 \
  --run output/eval_student_v1 --assessments output/metrics_student_v1/review_sheet.json \
  --out output/metrics_student_v1_reviewed
```

`resource_correct.rate` 的分母是实际人工审核条数，同时报告reviewed数量。unsupported_root_cause越低越好；其他指标不是一个统一总分。对证据不足的案例，合理答案可能是继续取证而非强行选择资源。

至少比较原始学生、微调学生、教师、量化学生；同一案例对照错在哪里。loss低但资源定位没进步时优先查标签、数据覆盖和输入缺口。量化后退化要重新选择精度，不只看模型文件变小。

验收：报告实际测试条数、每语种覆盖、人工评测覆盖、关键错误、完整耗时。每天200条若在8小时内串行处理，平均每条约144秒预算；音频数不一定等于诊断请求数，还要计入多轮查询和人工复核。
