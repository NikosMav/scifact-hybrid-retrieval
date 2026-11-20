"""Independently rescore published SciFact-Open BM25 and MiniLM candidates."""

import argparse
import json
from pathlib import Path

from scripts.benchmark_scifact_open import digest, hit_counts
from evidence_retrieval.eval.rationales import records
from evidence_retrieval.index import reciprocal_rank_fusion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rankings", type=Path, required=True)
    parser.add_argument("--claims", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    runs = json.loads(args.rankings.read_text())
    claims = records(args.claims)
    keys = {str(c["id"]) for c in claims}
    if runs["dataset"] != "scifact_open" or keys != runs["bm25"].keys() or keys != runs["dense"].keys():
        raise ValueError("Expected one BM25 and one MiniLM ranking for each SciFact-Open claim")
    predictions = []
    for claim in claims:
        lists = []
        for method in ["bm25", "dense"]:
            ranking = runs[method][str(claim["id"])]
            ids = [int(hit["doc_id"]) for hit in ranking]
            if len(set(ids)) != 10 or len(ids) != 10:
                raise ValueError("Expected ten unique candidates per method")
            lists.append([(int(hit["doc_id"]), hit["score"]) for hit in ranking])
        fused = reciprocal_rank_fusion(lists, fetch=10)
        predictions.append({"claim_id": claim["id"],
                            **{name: [str(i) for i, _ in ranks]
                               for name, ranks in zip(("bm25", "dense", "hybrid"), [*lists, fused])}})
    metrics = {"rankings_sha256": digest(args.rankings), "claims_sha256": digest(args.claims)}
    for name in ("bm25", "dense", "hybrid"):
        rows = [{"claim_id": row["claim_id"], "docs": row[name]} for row in predictions]
        metrics[name] = hit_counts(rows, claims)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "replay.jsonl").write_text("".join(json.dumps(r) + "\n" for r in predictions))
    (args.out / "replay_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
