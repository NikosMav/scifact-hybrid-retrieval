"""Measure BM25 and MiniLM RRF on identical SciFact-Open queries and judgments."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from platform import machine, python_version
from statistics import median
from time import perf_counter

import numpy as np

from evidence_retrieval.encoders import BM25Encoder
from evidence_retrieval.eval.rationales import records
from evidence_retrieval.index import reciprocal_rank_fusion


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class MiniLM:
    """Pinned fp32 all-MiniLM-L6-v2 export, masked mean pooling and L2 norm."""

    def __init__(self, model_dir, threads):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        expected = {
            "model.onnx": "759c3cd2b7fe7e93933ad23c4c9181b7396442a2ed746ec7c1d46192c469c46e",
            "tokenizer.json": "da0e79933b9ed51798a3ae27893d3c5fa4a201126cef75586296df9b4d2c62a0",
        }
        for name, sha in expected.items():
            if digest(model_dir / name) != sha:
                raise ValueError(f"Unexpected SHA-256 for {name}")
        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=256)
        self.tokenizer.enable_padding()
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        self.model = ort.InferenceSession(str(model_dir / "model.onnx"), options,
                                          providers=["CPUExecutionProvider"])

    def encode(self, texts):
        batch = self.tokenizer.encode_batch(texts)
        mask = np.asarray([x.attention_mask for x in batch], dtype=np.int64)
        inputs = {"input_ids": np.array([x.ids for x in batch], dtype=np.int64),
                  "attention_mask": mask, "token_type_ids": np.zeros_like(mask)}
        hidden = self.model.run(["last_hidden_state"], inputs)[0]
        vectors = (hidden * mask[:, :, None].astype(np.float32)).sum(axis=1)
        return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)


def relevance(claim, provenance):
    return {doc for doc, entry in claim["evidence"].items()
            if entry["label"] in {"SUPPORT", "CONTRADICT"}
            and (provenance == "all" or entry["provenance"] == provenance)}


def evaluate(ranker, ids, claims):
    rows, elapsed = [], []
    for claim in claims:
        begin = perf_counter()
        ranks = ranker(claim["claim"])
        elapsed.append(perf_counter() - begin)
        docs = [ids[i] for i, _ in ranks[:10]]
        rows.append({"claim_id": claim["id"], "docs": docs})
    return rows, {"hit_at_10": hit_counts(rows, claims), "median_query_ms": 1000 * median(elapsed)}


def hit_counts(rows, claims):
    if [r["claim_id"] for r in rows] != [c["id"] for c in claims]:
        raise ValueError("Rankings and judgments must have matching claim IDs in order")
    metrics = {}
    for source in ["all", "citation", "pooling"]:
        hit = total = 0
        for claim, row in zip(claims, rows):
            gold = relevance(claim, source)
            if gold:
                total += 1
                hit += bool(gold.intersection(row["docs"]))
        metrics[source] = {"hits": hit, "judged_claims": total}
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--claims", type=Path, required=True)
    parser.add_argument("--collection", choices=["original", "open"], required=True)
    parser.add_argument("--model-dir", type=Path, help="Include hybrid using pinned MiniLM ONNX files")
    parser.add_argument("--out", type=Path, required=True, help="New run directory (includes indexes)")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    ids, passages = [], []
    with args.corpus.open() as source:
        for line in source:
            doc = json.loads(line)
            ids.append(str(doc["doc_id"]))
            passages.append(" ".join([doc["title"], *doc["abstract"]]))
    if args.collection == "open" and len(ids) != 500_000:
        raise ValueError("Expected exactly 500,000 documents in the full SciFact-Open corpus")
    if args.collection == "original" and len(ids) != 5183:
        raise ValueError("Expected 5,183 documents in the original SciFact release")
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate document IDs")
    claims = records(args.claims)
    known = set(ids)
    if args.collection == "open" and any(relevance(c, "all") - known for c in claims):
        raise ValueError("Some judged abstracts are missing from the full corpus")
    args.out.mkdir(parents=True, exist_ok=False)
    lookup = args.out / "document_ids.json"
    lookup.write_text(json.dumps(ids))
    report = {"collection": args.collection, "documents": len(ids), "queries": len(claims),
              "corpus_sha256": digest(args.corpus), "claims_sha256": digest(args.claims),
              "architecture": machine(), "python": python_version(), "onnx_threads": args.threads,
              "blas_threads_env": os.getenv("OPENBLAS_NUM_THREADS")}

    start = perf_counter()
    bm25 = BM25Encoder().fit(passages)
    index_dir = bm25.save(args.out / "bm25")
    build = perf_counter() - start
    index_bytes = lookup.stat().st_size + sum(p.stat().st_size for p in index_dir.rglob("*") if p.is_file())
    def record(name, ranker, seconds, size):
        rows, metrics = evaluate(ranker, ids, claims)
        report[name] = dict(**metrics, build_seconds=seconds, index_bytes=size)
        (args.out / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        (args.out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
        print(name, report[name], flush=True)

    record("bm25", lambda text: bm25.rank(text, 10), build, index_bytes)
    if args.model_dir:
        start = perf_counter()
        encoder = MiniLM(args.model_dir, args.threads)
        vector_path = args.out / "minilm.npy"
        vectors = np.lib.format.open_memmap(vector_path, "w+", dtype="float32", shape=(len(ids), 384))
        for offset in range(0, len(ids), 16):
            vectors[offset:offset + 16] = encoder.encode(passages[offset:offset + 16])
        vectors.flush()
        hybrid_build = build + perf_counter() - start

        def hybrid(text):
            lexical = bm25.rank(text, 100)
            scores = vectors @ encoder.encode([text])[0]
            head = np.argsort(-scores, kind="stable")[:100]
            dense = [(int(i), scores[i]) for i in head]
            return reciprocal_rank_fusion([lexical, dense], fetch=10)

        record("hybrid", hybrid, hybrid_build, index_bytes + vector_path.stat().st_size)


if __name__ == "__main__":
    main()
