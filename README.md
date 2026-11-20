# SciFact Hybrid Retrieval

**Can a search engine show a reader just one evidence sentence?** On SciFact
development claims, a selective reader showed one sentence on **40/300**
claims. **28/40** matched the expert annotations. With a larger reading
budget, three selected sentences per retrieved abstract recovered a complete
rationale for **153/188** evidence-bearing claims. [Try the reader and inspect the tradeoffs](docs/reader.md).

On published rankings from **500,000 abstracts**, combining BM25 and MiniLM
reached **157/206** claims at rank ten. BM25 reached **134/206**. MiniLM alone
also reached **157/206**, with different successes. [SciFact-Open results and reproduction](docs/scifact-open.md).

## Can search find the sentence that answers a scientific claim?

BM25 put an evidence-bearing abstract in the top ten for **178 of 188 claims
(94.7%)** on the original SciFact development split. Given those hits, a
TF-IDF selector found an annotated sentence for **99 of 178 (55.6%)**.
Taking the first sentence of each retrieved abstract found one for **9 of 178
(5.1%)**. The larger opportunity is inside the paper: **79 sentence misses**
versus **10 abstract misses**.

For example, claim 5 puts abnormal prion protein prevalence at one in 2,000.
BM25 ranked the right paper first. The selector chose the sentence describing
the 32,441 samples. The annotated sentence gives the result: **493 per
million**, approximately one in 2,000.
[Inspect the hits, misses and original sentences](results/rationales/README.md).

The selector returns one sentence from each of ten retrieved abstracts. A
claim earns credit if any selection matches an annotated sentence in its own
abstract. That yields **99/188 (52.7%)** end to end. Requiring every sentence
of at least one annotated rationale gives **94/188 (50.0%)**. The other 112
development claims have no annotated rationale and are excluded. Support and
contradiction labels come from the annotators. The system retrieves evidence.

### Reproduce the evidence benchmark

From an editable install with the repository's CI dependencies:

```bash
python scripts/download_scifact.py /tmp/scifact-data
python scripts/run_rationale_eval.py --data /tmp/scifact-data --split dev --out /tmp/scifact-dev
```

The downloader verifies the release SHA-256. BM25 indexes titles and abstracts
with the settings below. The sentence selector fits sublinear word TF-IDF on
the corpus sentences, ranks by cosine similarity within each candidate
abstract and breaks ties by original sentence order. Neither stage uses the
evidence labels to rank. The report pairs examples with their published
rationales. [Per-claim predictions](results/rationales/claims.jsonl) include
all ten candidates, zero-based sentence positions and gold rationale groups.
[Dataset and annotation format](https://github.com/allenai/scifact/blob/master/doc/data.md).

## Document-ranking benchmark

On SciFact test (BEIR), BM25 reaches nDCG@10 **0.6762**, next to published
Anserini flat BM25 **0.6789** and BEIR BM25 **0.665**. Hybrid RRF reaches
**0.7194** and beats BM25 on a paired sign test (p=4.3×10⁻⁸; 74 up / 21 down /
205 ties). MiniLM alone does not beat BM25 (0.6451, p=0.73). Cross-encoder
reranking of hybrid's top 50 hurts (0.7194 → 0.6903) and is not significant
versus BM25 (p=0.10).

This is **not a fact-checker**. A retrieved abstract does not establish whether
the claim is true.

## SciFact test

`beir/scifact/test` via `ir_datasets`: 300 claims, 5,183 abstracts, one document
per abstract. Metrics are pytrec_eval nDCG@10, Recall@100, and reciprocal rank.
BM25 is `bm25s` with Lucene scoring, k1=0.9, b=0.4, English stopwords, and a
Snowball stemmer. Dense retrieval is `sentence-transformers/all-MiniLM-L6-v2`.
Hybrid is RRF (k=60) of those two rank lists. The word-window chunker is not used.

The train split (809 queries) was run first. The test split was scored only
after that metric code was frozen. Regenerate with:

```bash
pip install -e ".[scifact]"
python scripts/run_scifact_eval.py --split train
python scripts/run_scifact_eval.py --split test
```

<!-- SCIFACT_TEST_TABLE_START -->
| Method | nDCG@10 | Recall@100 | Reciprocal rank |
| --- | ---: | ---: | ---: |
| bm25 | 0.6762 | 0.9127 | 0.6456 |
| dense | 0.6451 | 0.9250 | 0.6110 |
| hybrid | 0.7194 | 0.9550 | 0.6869 |
<!-- SCIFACT_TEST_TABLE_END -->

BM25 nDCG@10 is 0.6762. The published anchors are BEIR BM25 **0.665** and
Anserini flat BM25 **0.6789**. Recall@100 for this BM25 run is 0.9127; BEIR
reports 0.908 for BM25.

MiniLM does not beat BM25 on nDCG@10 (0.6451 vs 0.6762; paired sign test
p=0.73). Hybrid does (0.7194; 74 queries up, 21 down, 205 ties, p=4.3×10⁻⁸).

### Ablations on the same test qrels

The reranker uses hybrid as its first stage because that system had the higher
train nDCG@10 (0.719 vs BM25 0.694 vs MiniLM 0.660). The test split was not
used to choose it.

<!-- SCIFACT_ABLATION_TABLE_START -->
| Method | nDCG@10 | Recall@100 | Reciprocal rank |
| --- | ---: | ---: | ---: |
| hybrid_rerank | 0.6903 | 0.9550 | 0.6652 |
| bge | 0.7127 | 0.9417 | 0.6866 |
| bge_hybrid | 0.7189 | 0.9650 | 0.6865 |
<!-- SCIFACT_ABLATION_TABLE_END -->

Replacing MiniLM with `BAAI/bge-small-en-v1.5` moves dense nDCG@10 from 0.6451
to 0.7127 (sign test against BM25, p=0.018). Queries use BGE's retrieval
instruction; abstracts do not. Putting that encoder into RRF does not move
hybrid nDCG@10 (0.7189 vs 0.7194). BGE hybrid Recall@100 does move, to 0.9650.

Reranking the top 50 hybrid hits with `cross-encoder/ms-marco-MiniLM-L-6-v2`
does not help. nDCG@10 goes from 0.7194 to 0.6903, and the comparison with BM25
is not significant (p=0.10). Recall@100 stays 0.9550 because the reranker only
reorders those 50 hits.

`sentence-transformers` stays `>=2.6,<4`. Both ablations use the existing
`SentenceTransformer.encode` and `CrossEncoder` APIs. Regenerate the ablation
rows with `python scripts/run_scifact_ablations.py --split test`.

Three misses from the frozen three-way run are in
[`results/scifact_failures.md`](results/scifact_failures.md): a methionine-restriction
claim whose BM25 hit is a lifespan abstract while MiniLM returns a miRNA review;
a vCJD prevalence claim that BM25 ranks first and MiniLM replaces with a different
case report; and a low-birth-weight claim that none of the three systems place
in the top 10.

On the same per-query file, 10 queries drop by at least 0.5 nDCG@10 versus BM25.
Query 785 goes from 1.0 to 0.0 while Recall@100 stays 1 (reciprocal rank 1/18).
The low-birth-weight claim is outside the top 10 and inside the top 100 for
MiniLM and hybrid. Hybrid loses Recall@100 to BM25 on 0 queries and is worse
than both parents on 0 test queries.

CI does not download BEIR or these models.

## Quick start

The test suite uses synthetic data and mocked dense encoders. It does not
download ISOT, BEIR, or MiniLM:

```bash
git clone https://github.com/NikosMav/scifact-hybrid-retrieval.git
cd scifact-hybrid-retrieval
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
python -m evidence_retrieval -h
```

On Python 3.13, building `pytrec_eval` needs a C++ compiler and Python headers
(Debian/Ubuntu: `sudo apt install g++ python3-dev`).

## ISOT closed-corpus demo

The same package can index ISOT news CSVs for a local query demo. That demo is
not the score. Setup, caveats, and regenerate commands are in
[`docs/isot-demo.md`](docs/isot-demo.md). Title-recovery and paraphrase outputs
stay in [`results/`](results/README.md).

## Repository map

| Path | Purpose |
| --- | --- |
| `evidence_retrieval/` | Chunking, BM25, MiniLM, hybrid RRF, and the CLI |
| `tests/` | Unit tests (no BEIR, no torch) |
| `results/` | SciFact scores, failures, and ablations; ISOT demo files |
| `docs/isot-demo.md` | ISOT closed-corpus side demo |
| `scripts/run_scifact_eval.py` | BM25 vs MiniLM vs RRF on SciFact |
| `scripts/run_scifact_ablations.py` | Cross-encoder rerank and BGE-small on the frozen test split |
| `evidence_retrieval.ipynb` | Short ISOT query demo of the same package |

## Limitations

- The committed SciFact test table is the comparison. Regenerating it needs `pip install -e ".[scifact]"` and the commands above. Tokenizer and BM25 parameters are named in `results/scifact_eval_meta.json`; they are not a guaranteed match to Elasticsearch's 0.665.
- The dense index is an in-memory brute-force cosine index. A few thousand vectors do not need an ANN service.
- The ISOT demo is a closed historical corpus. Its judgments are same-article title recovery, not independent qrels, and source-bucket labels can encode outlet and style.
- No web evidence is fetched.

## Data and license

`scripts/download_data.py` downloads the
[ISOT Fake News Dataset](https://onlineacademiccommunity.uvic.ca/isot/2022/11/27/fake-news-detection-datasets/)
for the closed-corpus demo. SciFact is loaded on demand via `ir_datasets`
(`beir/scifact`, CC BY-NC 2.0). The large CSVs, BEIR cache, and generated
indexes are not part of Git.

Code is released under the [MIT License](LICENSE.md). Dataset use remains subject
to the source dataset's terms.
