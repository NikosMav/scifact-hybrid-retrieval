# ISOT closed-corpus demo

This is a side demo of the same `evidence_retrieval` package, not the
evaluation. The scored comparison is SciFact document ranking in the
[root README](../README.md).

The package can index the ISOT news CSVs and return ranked passages for a
local query. Title recovery (the query is the article title; the gold set is
that article's passages), the paraphrase rewrite, and the chunk-size ablations
live in [`results/`](../results/README.md) so they can be regenerated. Do not
cite them next to the BEIR anchor.

The demo's default sparse arm is TF-IDF, which is what those committed files
used. SciFact uses BM25 (`sparse_backend="bm25"`).

```bash
python scripts/download_data.py
python -m evidence_retrieval build
python -m evidence_retrieval query "Federal Reserve raises interest rates" --top-k 5
```

The first build downloads `sentence-transformers/all-MiniLM-L6-v2` and writes
the index under `data/retrieval_index/default/`. Passages are about 120 words
with 20-word overlap. `python -m evidence_retrieval eval` regenerates the
title-recovery files. It does not rewrite the SciFact table.
`evidence_retrieval.ipynb` is the same demo on a 200-article sample.

## Caveats

- The ISOT demo is a closed historical corpus. Its judgments are same-article
  title recovery, not independent qrels, and source-bucket labels can encode
  outlet and style.
- ISOT labels are source buckets, not claim-level truth.
- No web evidence is fetched.
- Dataset: [ISOT Fake News Dataset](https://onlineacademiccommunity.uvic.ca/isot/2022/11/27/fake-news-detection-datasets/).
