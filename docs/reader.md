# Read one sentence, or open the evidence set

For a claim about fine particulate air pollution and anxiety, the selective
reader surfaces sentence 12 of SciFact abstract 791050. It reports the observed
association. For claim 129 about citations of open-access papers, the reader
abstains. The underlying top sentence discusses the authors’ previous citation
records. A reader can use `--mode first` to inspect it.

From the repository root with an editable install:

```bash
python scripts/download_scifact.py /tmp/scifact-data
python -m evidence_retrieval read --data-dir /tmp/scifact-data --model results/reader/reader.json \
  'Exposure to fine particulate air pollution is relate to anxiety prevalence.'
```

The command defaults to selective display. Use `--mode ranked --budget 3`
for individually ranked sentences in abstract order, or `--mode window
--budget 3` for a contiguous passage. Both inspect BM25’s ten highest-ranked
abstracts. The original lexical and learned selectors remain available under
the `evidence` command. [Earlier selector comparison](sentence-selection.md).

## One sentence across all ten abstracts

| Display rule | Displayed / all claims | Annotated matches / displayed | Displayed on claims without an annotation |
| --- | ---: | ---: | ---: |
| Always show the highest-scoring sentence | 300/300 (100%) | 97/300 (32.3%) | 112/112 |
| Allow abstention | 40/300 (13.3%) | 28/40 (70.0%) | 5/112 |

Selective display produced 12 unmatched outputs and withheld a sentence on
260 claims. The exact 95% binomial interval for its observed precision is
53.5–83.4%. An unmatched sentence lacks a corresponding published annotation.
Abstention means the reader did not clear its score cutoff.

The reader fits a logistic regression on sentences from all ten BM25
candidates. It uses the earlier selector's six lexical and structural features
plus reciprocal abstract rank. Each sentence has a positive target only when
its claim and paper have a matching sentence annotation. All 809 training
claims entered a deterministic split. Claims linked by any cited or
evidence-bearing paper stayed together. We fit on 599 claims and selected
the highest-coverage threshold reaching 70% annotated precision on at least
ten displayed calibration claims. It matched 17/24 across 210 calibration
claims. We froze the coefficients and threshold before scoring the 300 dev
claims. This split has been used previously to evaluate the project.

## Reading cost for a complete rationale

A complete hit requires every sentence of one gold rationale in the same
abstract. The following results use all 188 dev claims with annotated
evidence, including abstract retrieval misses. Sentence precision counts
annotated selections across all retrieved abstracts.

| Sentences per abstract | Complete rationale | Annotated selections / sentences displayed | Extra displayed per net additional complete hit vs one |
| --- | ---: | ---: | ---: |
| Highest-scoring one | 112/188 (59.6%) | 122/1,880 (6.5%) | — |
| Highest-scoring three | 153/188 (81.4%) | 227/5,640 (4.0%) | 91.7 |
| Best contiguous three | 142/188 (75.5%) | 197/5,640 (3.5%) | 125.3 |

The ranked three include the top sentence and recover 41 additional complete
rationales. The contiguous window maximizes summed sentence scores and may
exclude the highest-scoring sentence. Among 178 claims with at least one
singleton gold rationale, the window gained 30 and lost four relative to
one sentence. The ranked three gained 37 and lost zero. These costs count
every displayed sentence, including sentences in the other candidate papers.

Ten claims require multiple sentences for every accepted rationale. Each
three-sentence method recovered four of those ten. For claim 137, about
vision screening in older adults, the gold rationale spans sentences 8
and 10 of abstract 26016929. Both three-sentence policies recover the pair.
Most of the 41 additional hits came from having more chances to select a
singleton rationale.

```bash
OPENBLAS_NUM_THREADS=1 python scripts/evaluate_reader.py --data /tmp/scifact-data --out /tmp/reader
```

Each prediction records the selected document and zero-based sentence
indices. The CLI displays one-based sentence numbers. [Frozen reader and per-claim results](../results/reader/).
