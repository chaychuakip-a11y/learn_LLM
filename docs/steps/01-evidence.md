# 01 看懂一张问题单、分支和 dump

目标：先人工读懂数据，再让模型学。输入是 case.json，输出是可回读的证据表。

```bash
python -m ticket_sft.debug --id de_nav_002
python -m ticket_sft.debug --id th_poi_001
python -m ticket_sft.sandbox --case data/example/cases/de_nav_002 --code data/example/gold/de_nav_002.py
```

终端里的 SUB 表示替换，DEL 表示漏识别，INS 表示多识别。当前对齐使用空格分词和字符串比较；泰语、中文等真实数据需要上游按引擎的分词/归一化口径处理。这里不能把示例算法直接当成正式WER统计。

按顺序阅读源码：`analysis.load_case` → `base_evidence` → `branch_table` → `dump_view`。再读 `data/example/gold/de_nav_002.py`：它对分支、目标文本和dump做具体计算，然后 `json.dumps` 输出证据。模型最后一轮只能根据这些证据解释。

## 四种容易误判的情况

| 观察 | 支持什么 | 还不能确认什么 |
| --- | --- | --- |
| LAB正确文本没进候选 | 继续查目标覆盖与候选生成 | 必然是某个词典漏词 |
| 文本在候选里但最终REC不同 | 查获胜分支和打分融合 | 声学必然错误 |
| 词在词典，路径不可达 | 继续查实际加载WFST/连接关系 | 词典存在就没有资源问题 |
| dump有state数字，映射未做 | 保存模型路径观察 | 数字已解释成音素真值 |

分支分数只能在可比较的尺度内计算差值。获胜分支优先使用显式 winner 标记。声学best path也是模型观察，需要标注或独立复听验证。没有dump必须记录缺失，不能写成“查过无问题”。

`analysis.read_evidence('evidence/dump.txt')` 能读取来源清单登记的文本附件，返回 path/sha256/text/truncated。文本超过24KB只提供片段；`truncated=true`时不能据此断言全文件没有某个词。

练习：复制德语示例到 `data/my/draft_001`，改动 `state_map_applied` 或 winner，预测 debug 会变成什么，再运行。保留原示例。只有真正做了映射才能在真实数据里把映射标记改为true。

验收：能说清“看到的现象、候选原因、下一步证据”三者。Debug入口是 `debug.trace_case`、脚本stdout和原始case.json。
