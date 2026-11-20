# Finding evidence papers in 500,000 abstracts

We applied reciprocal rank fusion (RRF, k=60) to published top-ten BM25 and
MiniLM runs over the entire SciFact-Open corpus. We scored the three rankings
using the same 279 claims and published evidence judgments. Hit@10 is the
fraction of claims with at least one judged evidence paper in the top ten.
The 73 claims with no evidence judgment are excluded from that denominator.

| Full-corpus ranking | Any evidence | Citation evidence | Pooled evidence |
| --- | ---: | ---: | ---: |
| BM25 | 134/206 (65.0%) | 108/190 | 43/81 |
| MiniLM | 157/206 (76.2%) | 132/190 | 52/81 |
| BM25 + MiniLM | 157/206 (76.2%) | 135/190 | 51/81 |

Fusion gained 25 claims and lost two against BM25. Against MiniLM it gained
14 and lost 14. The provenance columns overlap because a claim can have
evidence from both sources. Pooled labels are document judgments. Pooled
sentence highlights are model-generated, so the sentence experiments use
the original expert rationale annotations.

The source run log reports 500,000 documents and identifies the MiniLM model.
We downloaded its per-claim ranked lists, checked IDs and ranking depths,
computed the hybrid rankings and rescored all methods. The full-corpus
index build time, stored size and query latency for our own implementation
remain unmeasured. The official raw corpus download returned HTTP 403 in
the cloud environment. [Source run](https://github.com/pernillejorg/retrieval-effects-claim-verification/tree/787d7f3b0380880341b5846efe26806b9f1dd032/results/step3_retrieval/SciFact-Open/MiniLM) · [Judgment definitions](https://github.com/dwadden/scifact-open/blob/master/doc/data.md) · [Recomputed rankings and counts](../results/scifact_open/).

## Same claims against the 5,183 original abstracts

We built both indexes locally, issued all 279 queries and timed each query
with a resident index. The original corpus contains fewer possible evidence
papers, especially from pooling. All 206 judged claims still count.
These results give a smaller-corpus reference. Differences across the two
tables reflect corpus, BM25 implementation and fusion depth together.

| Local index | Any evidence | Citation | Pooled | Index size | Build time | Median query |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 182/206 | 175/190 | 22/81 | 4.71 MB | 0.64 s | 0.18 ms |
| BM25 + MiniLM | 185/206 | 179/190 | 19/81 | 12.67 MB | 138.13 s | 4.39 ms |

Sizes are decimal MB of index files and document IDs. Model weights and
source text are excluded. The hybrid uses the project's Lucene-style BM25,
fp32 MiniLM with 256-token truncation and normalized masked-mean pooling,
and the top 100 per index for RRF. The cloud run used four CPU cores, four
ONNX threads and one BLAS thread. Times include index construction and
serialization after loading the source text. Model loading contributes to
the hybrid build time. [Local run metadata and rankings](../results/scifact_open/original/).

## Reproduce the full-corpus ranking analysis

Run from the repository root. Both inputs are pinned to source commits.
The output includes their SHA-256 digests.

```bash
curl -fL https://raw.githubusercontent.com/Jay-G14/LMA-GroupProject/71f87ba620338f6f482750945153273831490e8e/data/raw/scifact_open/claims.jsonl -o /tmp/open-claims.jsonl
curl -fL https://raw.githubusercontent.com/pernillejorg/retrieval-effects-claim-verification/787d7f3b0380880341b5846efe26806b9f1dd032/results/step3_retrieval/SciFact-Open/MiniLM/retrieval_candidates_minilm_scifact_open.json -o /tmp/open-ranks.json
python -m scripts.rescore_open_rankings --rankings /tmp/open-ranks.json --claims /tmp/open-claims.jsonl --out /tmp/open-replay
```

## Build and time an index

Install the `open-benchmark` extra or add `onnxruntime` and `tokenizers` to
the CI environment. The following fetches the pinned fp32 ONNX model package.
The benchmark verifies model and tokenizer checksums before inference.

```bash
mkdir -p /tmp/minilm
(cd /tmp/minilm && npm pack --ignore-scripts @maycombhq/embed-model@2.0.0-beta.1 && tar xzf maycombhq-embed-model-2.0.0-beta.1.tgz)
python scripts/download_scifact.py /tmp/scifact-data
OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false python scripts/benchmark_scifact_open.py \
  --corpus /tmp/scifact-data/corpus.jsonl --claims /tmp/open-claims.jsonl --collection original \
  --model-dir /tmp/minilm/package --out /tmp/original-index
```

For the full local build, supply the official release's `data/corpus.jsonl`
and `data/claims.jsonl`, select `--collection open` and use a fresh output
directory. The runner requires 500,000 unique documents and checks gold
document coverage. Omit `--model-dir` for a BM25-only run.
[Official download script](https://github.com/dwadden/scifact-open/blob/master/script/get_data.sh).
