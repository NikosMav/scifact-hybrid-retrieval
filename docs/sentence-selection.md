# Which sentence would you show a reader?

A scientific abstract often repeats the hypothesis in its objective and resolves it in its results. SciFact claim 399 is an example. BM25 ranks the correct paper first. Both selectors see the same abstract and can show exactly one sentence.

| TF-IDF | Learned selector |
| --- | --- |
| “OBJECTIVE To determine whether higher past exposure to particulate air pollution is associated with prevalent high symptoms of anxiety.” | “CONCLUSIONS Exposure to fine particulate matter (PM2.5) was associated with high symptoms of anxiety, with more recent exposures potentially more relevant than more distant exposures.” |

The annotators marked the conclusion as evidence. The distinction matters here because a statement of the research question can match nearly every word in the claim. The study reports an association.

## Results on the original SciFact development split

| Measure | TF-IDF | Learned selector |
| --- | ---: | ---: |
| Evidence sentence given an abstract hit | 99/178 (55.6%) | 109/178 (61.2%) |
| Evidence sentence end to end | 99/188 (52.7%) | 109/188 (58.0%) |
| Complete annotated rationale end to end | 94/188 (50.0%) | 103/188 (54.8%) |
| Evidence sentence on claims whose gold papers never appear as training evidence | 43/76 (56.6%) | 41/76 (53.9%) |

The retriever ranks the same ten abstracts for each selector. Each selector returns one sentence per abstract, so a reader may have to inspect ten sentences. A claim succeeds if one selected sentence matches a published rationale in the corresponding abstract. A complete-rationale hit requires covering every sentence of at least one rationale. The single-sentence budget limits which rationales can be recovered. The 112 dev claims without annotated evidence are outside this comparison.

The learned selector gained 29 claims and lost 19, with 140 ties (two-sided paired sign test, p = 0.1934). Claim 129 illustrates a regression. The claim concerns whether open-access papers receive fewer citations. TF-IDF found the annotated sentence reporting that open-access papers were cited more. The learned selector picked a sentence about the authors’ *previous* citation records. This reversal and the unseen-paper result are reasons to retain TF-IDF as the default. The dev split had already been inspected in the baseline project. A private test result would need separate evaluation.

## Run the comparison or try a claim

Start in the editable installation described in the repository README. The downloader verifies the public release. The learned selector fits a logistic regression on the original training split, using sentences within evidence-bearing abstracts as examples. Positive sentences belong to at least one annotated rationale. Inputs are cosine similarity, similarity relative to the strongest sentence, relative position, a results/conclusions/findings heading, digit presence and log word count. Neither document IDs nor stance labels are features.

```bash
python scripts/download_scifact.py /tmp/scifact-data
python scripts/compare_sentence_selectors.py --data /tmp/scifact-data --out /tmp/comparison.csv
python -m evidence_retrieval evidence --data-dir /tmp/scifact-data --selector learned --top-k 1 \
  "Exposure to fine particulate air pollution is relate to anxiety prevalence."
```

Replace the quoted claim to search the full corpus. Use `--selector tfidf` for the lexical baseline. The CLI numbers sentences starting at one. The predictions CSV stores zero-based offsets in the original abstract, aligned with the ten document IDs in retrieval order. [Inspect all 188 paired predictions](../results/selector_comparison.csv).
