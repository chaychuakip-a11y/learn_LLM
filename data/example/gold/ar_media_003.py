import json
import analysis

case = analysis.load_case()
evidence = analysis.base_evidence(case)
evidence["branch_table_empty"] = evidence["branches"] == []
evidence["deletions"] = [
    {"utt_id": row["utt_id"], "tokens": row["deletions"]} for row in evidence["align"]
]
print(json.dumps(evidence, ensure_ascii=False, indent=2))
