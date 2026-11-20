"""Compare one sentence per BM25 candidate on the original SciFact dev split."""

import argparse
import csv
from pathlib import Path

from evidence_retrieval.eval.rationales import EvidenceRetriever, records, score_claim
from evidence_retrieval.eval.trec import paired_sign_test


def evaluate(data):
    retriever = EvidenceRetriever(records(data / "corpus.jsonl"))
    train = records(data / "claims_train.jsonl")
    dev = records(data / "claims_dev.jsonl")
    if {c["id"] for c in train} & {c["id"] for c in dev}:
        raise ValueError("Claim IDs overlap between train and dev")
    selector = retriever.fit_selector(train)
    seen = {doc_id for c in train for doc_id in c["evidence"]}
    rows = []
    for claim in dev:
        if not claim["evidence"]:
            continue
        base = score_claim(claim, retriever.retrieve(claim["claim"]))
        learned = score_claim(claim, retriever.retrieve(claim["claim"], selector=selector))
        base_ids = [h["doc_id"] for h in base["candidates"]]
        if base_ids != [h["doc_id"] for h in learned["candidates"]]:
            raise ValueError("Candidate abstracts differ across methods")
        rows.append({
            "claim_id": claim["id"], "claim": claim["claim"],
            "new_papers": int(seen.isdisjoint(claim["evidence"])),
            "abstract_hit": int(base["abstract_hit"]),
            "tfidf": int(base["sentence_hit"]), "learned": int(learned["sentence_hit"]),
            "tfidf_complete": int(base["complete_hit"]), "learned_complete": int(learned["complete_hit"]),
            "docs": " ".join(base_ids),
            "tfidf_sentence_ids": " ".join(str(h["sentence_id"]) for h in base["candidates"]),
            "learned_sentence_ids": " ".join(str(h["sentence_id"]) for h in learned["candidates"]),
        })
    return rows


def report(rows):
    print("| Measure | TF-IDF | Learned selector |")
    print("| --- | ---: | ---: |")
    def measure(label, subset, suffix=""):
        def cell(method):
            hits = sum(r[method + suffix] for r in subset)
            return f"{hits}/{len(subset)} ({hits/len(subset):.1%})" if subset else "n/a"
        print(f"| {label} | {cell('tfidf')} | {cell('learned')} |")

    measure("Evidence sentence given an abstract hit", [r for r in rows if r["abstract_hit"]])
    measure("Evidence sentence end to end", rows)
    measure("Complete rationale end to end", rows, "_complete")
    measure("Evidence sentence on claims with only unseen gold papers", [r for r in rows if r["new_papers"]])
    paired = paired_sign_test({r["claim_id"]: r["learned"] for r in rows},
                              {r["claim_id"]: r["tfidf"] for r in rows})
    print(f"\nPaired sentence hits: {paired['n_pos']} wins, {paired['n_neg']} losses, "
          f"{paired['n_ties']} ties. Two-sided sign test p={paired['pvalue']:.4g}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="Paired predictions CSV")
    args = parser.parse_args()
    rows = evaluate(args.data)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as dest:
        writer = csv.DictWriter(dest, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    report(rows)


if __name__ == "__main__":
    main()
