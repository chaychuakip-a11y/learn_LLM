import json
import analysis

case = analysis.load_case()
evidence = analysis.base_evidence(case)
for row in evidence["branches"]:
    matched = [item for item in row["candidates"] if item["equals_lab"]]
    if len(matched) != 1:
        raise ValueError(f"{row['utt_id']} 里等于标注的候选不是 1 条")
    row["winner_minus_target"] = round(row["winner_score"] - matched[0]["score"], 1)
winner_by_utt = {row["utt_id"]: row["winner_text"] for row in evidence["branches"]}
for item in evidence["dump"]:
    item["matches_winner_text"] = item["best_path_text"] == winner_by_utt.get(item["utt_id"])
print(json.dumps(evidence, ensure_ascii=False, indent=2))
