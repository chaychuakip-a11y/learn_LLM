# 02 准备真实数据与来源快照

一条问题单不是只放一句问答。你至少需要：原语种 LAB/最终REC、语种和车厂、引擎与资源版本、实际加载关系、可用dump/日志，以及后续实验结果。哪些材料没有取得，就明确留空。

按 `configs/case.template.json` 整理你的第一条案例。结构化字段由已有解析器或人工提供，snapshot不会猜私有dump格式。

```text
data/my/raw/ticket_001/
  case.json
  audio/u1.wav
  evidence/dump.txt
  evidence/runtime_resources.json
```

case.json 的 lexicon 是你所检查资源中的词，不等于所有资源；wfst 的路径状态必须来自实际查询。可在case中补充 resource_manifest，记录每份资源的类型、版本、分支、语言和加载关系。历史结案答案、教师结论、修复后的未来证据不得混入当时模型应看到的输入。

冻结来源：

```bash
python -m ticket_sft.snapshot \
  --case-dir data/my/raw/ticket_001 \
  --out data/my/snapshots/ticket_001 \
  --source evidence/dump.txt \
  --source evidence/runtime_resources.json \
  --original-audio audio/u1.wav \
  --group-id incident_001 --origin real
```

产物：新case.json、声明的附件、SHA256、原音频指纹。每次更新做新快照；程序不覆盖已有目录。

`group_id` 表示独立故障组。同一原音频的裁剪/转码、同一单多步骤、同一问题的近重复版本应放同组。原音频指纹能抓到字节相同的重复，不能自动识别所有转码副本，因此人工来源关联仍需要保留。

关键代码：`snapshot.snapshot`复制声明文件；`evidence.source_path`检查路径；`read_case`核验哈希。哈希正确只能说明文件没变，不说明标注正确。不要把合成样例改一个字段就当成真实业务样本。

练习：冻结一份样例附件后改动快照中的附件，再运行教师dry-run，应该遇到 `E_SOURCE_HASH`。真实快照不要用这种方式做练习，另建演练目录。

验收：你能按sources回读证据、区分原音频与派生音频、解释每个group。先整理20条独立真实案例做试点，别急着扩充大量未审问答。
