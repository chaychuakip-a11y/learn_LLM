import json
import analysis

case = analysis.load_case()
evidence = analysis.base_evidence(case)
slot_rows = []
for utterance in case["utterances"]:
    person = (case.get("slots") or {}).get(utterance["utt_id"], {}).get("person")
    rec_tokens = str(utterance["rec"]).split()
    slot_rows.append(
        {
            "utt_id": utterance["utt_id"],
            "person": person,
            "person_in_rec": bool(person) and person in rec_tokens,
            "person_in_lexicon": bool(person) and person in case["lexicon"],
        }
    )
evidence["slot_check"] = slot_rows
print(json.dumps(evidence, ensure_ascii=False, indent=2))
