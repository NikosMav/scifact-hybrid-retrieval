# Where to take SciFact evidence retrieval next

## Round 1 · Scientific evidence systems

| Project | What the authors achieved | Implication here |
| --- | --- | --- |
| VeriSci | Released the corpus, expert judgments and a pipeline with separate reproduction commands for rationale selection and label prediction. The original test labels are private. [Repository](https://github.com/allenai/scifact) | Keep sentence retrieval independently measurable. Any future stance classifier needs its own evaluation. |
| MultiVerS | Shared one representation of the claim and full document across rationale and label prediction. The authors report beating two baselines on three scientific verification datasets and release the checkpoints. [Repository and paper](https://github.com/dwadden/multivers) | A context-aware reader is a meaningful challenger on the same ten abstracts. Audit checkpoint training data before evaluating it on dev. |
| ARSJoint | Jointly learned abstract retrieval, rationale selection and stance. Its evaluation requires complete evidence sets and a correct stance for abstract-level credit, with up to three predicted sentences. [Evaluation protocol](https://github.com/ZhiweiZhang97/ARSJointModel#evaluation) | Compare one- and three-sentence budgets, reporting complete-rationale recovery and precision together. Our top-ten claim hit rates cannot be ranked against their verification F1. |

## Round 2 · Reproducible and larger-scale retrieval

| Project | What it provides | Implication here |
| --- | --- | --- |
| BEIR | A common evaluator for heterogeneous IR tasks, including SciFact, with ranked runs and document metrics at multiple cutoffs. [Repository](https://github.com/beir-cellar/beir) | Preserve document rankings in sentence experiments so a change in retrieval cannot masquerade as a better reader. |
| Pyserini | Prebuilt indexes, queries, judgments and commands to reproduce reference retrieval runs. [Reproduction guides](https://github.com/castorini/pyserini/blob/master/docs/reproducibility.md) | Ship per-claim predictions alongside aggregate results and a command that regenerates them. The paired selector comparison does both. |
| SciFact-Open | Expanded to 500,000 abstracts with additional judgments collected by pooling. For pooled documents the supplied sentence highlights are model predictions. [Data documentation](https://github.com/dwadden/scifact-open/blob/master/doc/data.md) | Use pooled judgments to test abstract retrieval at scale. Use expert-annotated citation evidence for the sentence benchmark. |

## Experiments in priority order

1. **Rank sentences across the ten candidates and permit abstention.** The current success metric allows ten displayed sentences and excludes 112 claims without annotated evidence. Choose a cutoff on training folds grouped by evidence paper. Evaluate precision of the first displayed sentence, coverage, and how often the system returns a sentence for a dev claim without an annotated rationale. Treat those last cases as unjudged outside the annotations.
2. **Recover multi-sentence rationales.** Compare adjacent sentence windows against a context-aware reader with one and three sentences per candidate. Keep BM25 fixed. Report the extra sentences read per additional complete rationale, including losses on single-sentence claims. The current learned selector recovers a complete rationale for 103/188 claims at the one-sentence budget.
3. **Test retrieval as the corpus grows.** Run BM25 and hybrid retrieval on the full SciFact-Open corpus. Report abstract Hit@10, index size, build time and median query latency on the same hardware. Keep pooled and citation-derived judgments separate. The 94.7% abstract hit rate on the original evidence-bearing dev subset is a starting measurement for that experiment.

The current release implements a CPU selector, an arbitrary-claim search command and a paired comparison. The largest uncertainty for the next model is generalization to new papers. On the 76 claims backed entirely by papers absent from training evidence, TF-IDF found the annotated sentence on two more claims. [Comparison and worked examples](sentence-selection.md).
