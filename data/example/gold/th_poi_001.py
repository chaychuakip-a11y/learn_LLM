import json
import analysis

case = analysis.load_case()
evidence = analysis.base_evidence(case)
path_gap = []
lexicon = set(case["lexicon"])
for utt_id, info in (case.get("wfst") or {}).items():
    missing = info.get("missing") or []
    path_gap.append(
        {
            "utt_id": utt_id,
            "path_complete": info.get("path_complete"),
            "missing": missing,
            "in_lexicon_but_path_blocked": [word for word in missing if word in lexicon],
        }
    )
evidence["path_gap"] = path_gap
print(json.dumps(evidence, ensure_ascii=False, indent=2))
