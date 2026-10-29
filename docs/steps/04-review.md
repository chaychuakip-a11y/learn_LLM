# 04 审核、分组和构建训练数据

先读每个候选的 analysis.py、stdout.json、conclusion.md、原始附件，区分观察、假设、已验证结论。规则检查只覆盖有限格式与典型越界，不能代替技术审核。

决定三份集合：train用于更新参数；val用于调参数；test在方案确定后评估。建议按独立问题组结合车厂、语种、时间划分，约80/10/10可以作为起点。小规模试点必须报告每种语言实际数量，不能只看比例。

选定或重做一个正确候选后记录审核：

```bash
python -m ticket_sft.intranet approve \
  --candidate output/teachers/ds_v1/ticket_001 \
  --dataset data/my/reviewed_v1 \
  --reviewer your_name --group-id incident_001 --split train \
  --note '已对照实际加载资源和分支日志；声学映射仍缺失，保留待核实'
```

同一case_id不能批准两位教师的答案各一份。直接手改已生成的候选会触发产物哈希变化。需要人工修订时，把脚本和结论复制到草稿目录编辑，再生成一份新候选：

```bash
python -m ticket_sft.intranet curate --case data/my/snapshots/ticket_001 \
  --code data/my/drafts/ticket_001.py --conclusion data/my/drafts/ticket_001.md \
  --source-candidate output/teachers/ds_v1/ticket_001 \
  --out output/curated/ticket_001_v2
```

curate不调用模型，会实际重放修订脚本，重新检查结论，保留原教师候选。对新目录再次执行approve；人工修订也不自动批准。

已审核目录：

```text
data/my/reviewed_v1/cases/ticket_001/case.json
data/my/reviewed_v1/cases/ticket_001/evidence/...
data/my/reviewed_v1/gold/ticket_001.py
data/my/reviewed_v1/gold/ticket_001.md
data/my/reviewed_v1/gold/ticket_001.meta.json
```

至少准备各集合一个独立案例，然后构建：

```bash
python -m ticket_sft.intranet build --dataset data/my/reviewed_v1 --out data/my/dataset_v1
```

阅读 `dataset.build_reviewed`。它先核对审核状态与文件哈希，检查同group/同原音频不跨集合，重放脚本，最后才写新目录。

产物 `swift/train.jsonl` 保持原仓库格式；`gemma/train.jsonl`、`valid.jsonl`、`test.jsonl` 使用两条决策：第一条学脚本，第二条读真实工具结果学结论。每个case两步必须同组，工具stdout不当作预测目标。

split_report.json列出实际case/group/语言/教师/哈希，作为版本记录。以后追加样本时已有组保持原split；所有近重复、同音频变体跟随原组。不要通过把测试失败项搬进训练来提高旧测试成绩，应发布新的测试协议和数据版本。

Debug：E_UNREVIEWED查meta；E_REVIEW_STALE查审核后改动；E_LEAKAGE查group和音频来源；E_EMPTY_SPLIT补充数据。练习：在演练副本故意让同一音频跨集合，确认程序拒绝。
