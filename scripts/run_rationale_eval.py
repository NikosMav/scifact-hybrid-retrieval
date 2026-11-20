"""Evaluate the original SciFact annotations and write an inspectable report."""

import argparse
import json
from pathlib import Path

from evidence_retrieval.eval.rationales import EvidenceRetriever, records, score_claim


def fraction(n, d):
    return f"{n}/{d} ({n/d:.1%})" if d else "0/0 (undefined)"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "dev"], default="dev")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    corpus = records(args.data / "corpus.jsonl")
    documents = {str(doc["doc_id"]): doc for doc in corpus}
    claims = records(args.data / f"claims_{args.split}.jsonl")
    retriever = EvidenceRetriever(corpus)
    rows = []
    for claim in claims:
        if claim["evidence"]:
            rows.append(score_claim(claim, retriever.retrieve(claim["claim"])))
    if not rows:
        raise ValueError("No annotated evidence in this split")
    found = sum(row["abstract_hit"] for row in rows)
    selected = sum(row["sentence_hit"] for row in rows)
    complete = sum(row["complete_hit"] for row in rows)
    lead = sum(row["lead_hit"] for row in rows)
    report = [
        f"# Finding scientific evidence · SciFact {args.split}", "",
        f"{len(corpus):,} abstracts · {len(rows)} claims with evidence · "
        f"{len(claims) - len(rows)} claims without annotated evidence excluded", "",
        "| Question | Claims |", "| --- | ---: |",
        f"| Evidence abstract in BM25 top 10? | {fraction(found, len(rows))} |",
        f"| Selected an evidence sentence, given an abstract hit? | {fraction(selected, found)} |",
        f"| First sentence of each abstract, given an abstract hit? | {fraction(lead, found)} |",
        f"| Selected an evidence sentence end to end? | {fraction(selected, len(rows))} |",
        f"| Recovered a complete rationale end to end? | {fraction(complete, len(rows))} |",
        "", f"BM25 missed the abstract on {len(rows)-found} claims. "
        f"The sentence selector missed on {found-selected} more.", "",
    ]
    for name, abstract, sentence in [("Sentence found", True, True),
                                      ("Sentence missed", True, False),
                                      ("Abstract missed", False, False)]:
        row = next((r for r in rows if r["abstract_hit"] == abstract and
                    r["sentence_hit"] == sentence), None)
        if row is None:
            continue
        report.extend([f"## {name}", "", f"Claim {row['claim_id']} · {row['claim']}", ""])
        hit = next((m for m in row["matches"] if m["sentence_hit"]), None)
        if hit is None:
            hit = (row["matches"] or row["candidates"])[0]
        rank = next(i for i, c in enumerate(row["candidates"], 1) if c["doc_id"] == hit["doc_id"])
        report.extend([f"Retrieved at rank {rank}: "
                       f"{hit['title']}", "", f"> {hit['sentence'].strip()}", ""])
        claim = next(c for c in claims if c["id"] == row["claim_id"])
        doc_id = hit["doc_id"] if abstract else next(iter(claim["evidence"]))
        groups = claim["evidence"][doc_id]
        group = next((g for g in groups if sentence and hit["sentence_id"] in g["sentences"]), groups[0])
        report.extend([f"Annotated rationale · {group['label']} · {documents[doc_id]['title']}", ""])
        report.extend("> " + documents[doc_id]["abstract"][i].strip() for i in group["sentences"])
        report.append("")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "README.md").write_text("\n".join(report).rstrip() + "\n", encoding="utf-8")
    with (args.out / "claims.jsonl").open("w", encoding="utf-8") as dest:
        for row in rows:
            dest.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("\n".join(report[:11]))


if __name__ == "__main__":
    main()
