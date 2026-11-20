"""Train the budgeted reader, freeze its cutoff, then evaluate SciFact dev."""

import argparse
import json
from pathlib import Path

from evidence_retrieval.eval.rationales import EvidenceRetriever, records
from evidence_retrieval.reader import (
    Reader, calibrate, candidates, first_sentence, is_evidence, select_budget, training_split,
)


def evaluate_claim(reader, retriever, claim):
    scored = reader.score(candidates(retriever, claim["claim"]))
    first = first_sentence(scored)
    result = {"claim_id": claim["id"], "has_evidence": bool(claim["evidence"]), "first": None}
    if first:
        doc, pos, score = first
        result["first"] = {"doc": str(doc["doc_id"]), "sentence": pos, "score": score,
                           "hit": is_evidence(claim, doc, pos), "shown": reader.accepts(score)}
    result["documents"] = []
    for doc, scores in scored:
        picks = {name: select_budget(scores, budget, adjacent)
                 for name, budget, adjacent in [("one", 1, False), ("three", 3, False),
                                                 ("window", 3, True)]}
        result["documents"].append({"doc": str(doc["doc_id"]), **picks})
    return result


def summarize(rows, claims):
    counts = {}
    judged = [r for r in rows if r["has_evidence"]]
    for policy in ("one", "three", "window"):
        read = correct = recovered = single_misses = 0
        for r in judged:
            gold = claims[r["claim_id"]]["evidence"]
            complete = False
            for d in r["documents"]:
                selected = set(d[policy])
                read += len(selected)
                groups = gold.get(d["doc"], [])
                positive = {i for g in groups for i in g["sentences"]}
                correct += len(selected & positive)
                complete |= any(set(g["sentences"]) <= selected for g in groups)
            recovered += complete
            if any(len(g["sentences"]) == 1 for groups in gold.values() for g in groups):
                single_misses += not complete
        counts[policy] = dict(read=read, annotated_sentences=correct, complete=recovered,
                              misses_on_claims_with_singleton=single_misses)
    for key, selected in [("always", [r for r in rows if r["first"]]),
                          ("selective", [r for r in rows if r["first"] and r["first"]["shown"]])]:
        counts[key] = {"shown": len(selected), "correct": sum(r["first"]["hit"] for r in selected),
                       "shown_without_annotation": sum(not r["has_evidence"] for r in selected)}
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    retriever = EvidenceRetriever(records(args.data / "corpus.jsonl"))
    fit, calibration = training_split(records(args.data / "claims_train.jsonl"))
    reader = Reader.fit(retriever, fit)
    observations = []
    for claim in calibration:
        first = first_sentence(reader.score(candidates(retriever, claim["claim"])))
        if first:
            doc, pos, score = first
            observations.append((score, is_evidence(claim, doc, pos)))
    reader.threshold = calibrate(observations)
    args.out.mkdir(parents=True, exist_ok=True)
    reader.save(args.out / "reader.json")
    dev = records(args.data / "claims_dev.jsonl")
    rows = [evaluate_claim(reader, retriever, c) for c in dev]
    summary = summarize(rows, {c["id"]: c for c in dev})
    summary.update(fit=len(fit), calibration=len(calibration), calibration_shown=sum(
        reader.accepts(s) for s, _ in observations), calibration_correct=sum(
            reader.accepts(s) and hit for s, hit in observations), dev=len(dev),
        judged=sum(r["has_evidence"] for r in rows))
    with (args.out / "dev.jsonl").open("w") as dest:
        for row in rows:
            print(json.dumps(row, allow_nan=False), file=dest)
    output = json.dumps(summary, indent=2) + "\n"
    (args.out / "metrics.json").write_text(output)
    print(output)


if __name__ == "__main__":
    main()
