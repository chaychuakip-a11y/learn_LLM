"""案例上的只读分析函数。

示例文本用空格分词，标注和候选按整串精确匹配。
真实引擎的分词、大小写和标点归一化，应在写成 case.json 之前做完。
"""

from __future__ import annotations

import json
import os
from difflib import SequenceMatcher
from pathlib import Path


def load_case(case_dir: str | os.PathLike[str] | None = None) -> dict:
    directory = Path(case_dir or os.environ["CASE_DIR"]).resolve()
    path = directory / "case.json"
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} 的顶层必须是对象")
    return data


def align_lab_rec(utterances: list[dict]) -> list[dict]:
    rows = []
    for utterance in utterances:
        lab = str(utterance["lab"]).split()
        rec = str(utterance["rec"]).split()
        substitutions, deletions, insertions = _diff_tokens(lab, rec)
        rows.append(
            {
                "utt_id": utterance["utt_id"],
                "lab": utterance["lab"],
                "rec": utterance["rec"],
                "substitutions": substitutions,
                "deletions": deletions,
                "insertions": insertions,
            }
        )
    return rows


def lexicon_check(utterances: list[dict], lexicon: list[str]) -> list[dict]:
    known = set(lexicon)
    rows = []
    for utterance in utterances:
        missing = [token for token in str(utterance["lab"]).split() if token not in known]
        rows.append({"utt_id": utterance["utt_id"], "missing": missing})
    return rows


def branch_table(pgs_rlt: list[dict], utterances: list[dict]) -> list[dict]:
    lab_by_id = {item["utt_id"]: item["lab"] for item in utterances}
    rows = []
    for item in pgs_rlt:
        utt_id = item["utt_id"]
        candidates = []
        for branch in item.get("branches") or []:
            text = branch["text"]
            candidates.append(
                {
                    "name": branch["name"],
                    "text": text,
                    "score": branch["score"],
                    "winner": bool(branch["winner"]),
                    "equals_lab": text == lab_by_id.get(utt_id),
                }
            )
        winners = [candidate for candidate in candidates if candidate["winner"]]
        if len(winners) != 1:
            raise ValueError(f"{utt_id} 应该恰好有一个获胜分支，现在有 {len(winners)} 个")
        winner = winners[0]
        rows.append(
            {
                "utt_id": utt_id,
                "target_in_candidates": any(candidate["equals_lab"] for candidate in candidates),
                "winner": winner["name"],
                "winner_text": winner["text"],
                "winner_score": winner["score"],
                "candidates": candidates,
            }
        )
    return rows


def dump_view(case: dict) -> list[dict]:
    dump = case.get("acoustic_dump")
    if not dump:
        return []
    rows = []
    for utt_id, info in dump.items():
        rows.append(
            {
                "utt_id": utt_id,
                "best_path_text": info.get("best_path_text"),
                "state_ids": info.get("state_ids") or [],
                "state_map_applied": bool(info.get("state_map_applied")),
            }
        )
    return rows


def base_evidence(case: dict) -> dict:
    return {
        "case_id": case["case_id"],
        "oem": case["oem"],
        "language": case["language"],
        "runtime": case.get("runtime"),
        "score_type": case.get("score_type"),
        "version": case.get("version"),
        "resource_slot": case.get("resource_slot"),
        "test_set": case.get("test_set"),
        "dump_present": case.get("acoustic_dump") is not None,
        "align": align_lab_rec(case["utterances"]),
        "lexicon_miss": lexicon_check(case["utterances"], case["lexicon"]),
        "branches": branch_table(case.get("pgs_rlt") or [], case["utterances"]),
        "wfst": case.get("wfst") or {},
        "slots": case.get("slots") or {},
        "dump": dump_view(case),
    }


def _diff_tokens(lab: list[str], rec: list[str]) -> tuple[list[dict], list[str], list[str]]:
    substitutions: list[dict] = []
    deletions: list[str] = []
    insertions: list[str] = []
    matcher = SequenceMatcher(a=lab, b=rec)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if tag == "delete":
            deletions.extend(lab[i1:i2])
            continue
        if tag == "insert":
            insertions.extend(rec[j1:j2])
            continue
        left = lab[i1:i2]
        right = rec[j1:j2]
        paired = max(len(left), len(right))
        for index in range(paired):
            lab_token = left[index] if index < len(left) else None
            rec_token = right[index] if index < len(right) else None
            if lab_token and rec_token:
                substitutions.append({"lab": lab_token, "rec": rec_token})
            elif lab_token:
                deletions.append(lab_token)
            elif rec_token:
                insertions.append(rec_token)
    return substitutions, deletions, insertions
