import numpy as np

from evidence_retrieval.reader import calibrate, first_sentence, select_budget, training_split


def test_threshold_uses_all_score_ties_and_handles_unattainable_target():
    observations = [(2., True), (1., False), (1., True)]
    assert calibrate(observations, target=0.75, minimum=1) == 2.
    assert calibrate(observations, target=0.75, minimum=2) is None


def test_group_split_keeps_transitively_linked_citations_together():
    docs = [[1, 2], [3, 4], [2, 3], [5], [6], [7], [8]]
    claims = [{"id": i, "cited_doc_ids": group, "evidence": {}} for i, group in enumerate(docs)]
    a, b = training_split(claims)
    assert {d for c in a for d in c["cited_doc_ids"]}.isdisjoint(d for c in b for d in c["cited_doc_ids"])
    assert len(a) + len(b) == len(claims)


def test_global_first_ties_and_non_contiguous_rationales():
    a, b = {"doc_id": 1}, {"doc_id": 2}
    assert first_sentence([(a, np.array([1., 2.])), (b, np.array([2.]))]) == (a, 1, 2.)
    scores = np.array([4., 0., 2., 3.])
    assert select_budget(scores, 3, adjacent=False) == [0, 2, 3]
    assert select_budget(scores, 3, adjacent=True) == [0, 1, 2]
    assert select_budget(scores[:1], 3, adjacent=True) == [0]


def test_open_hit_at_ten_counts_claims_once_and_separates_judgment_sources():
    from scripts.benchmark_scifact_open import hit_counts

    claims = [{"id": 1, "evidence": {"a": {"label": "SUPPORT", "provenance": "citation"},
                                      "b": {"label": "CONTRADICT", "provenance": "pooling"}}},
              {"id": 2, "evidence": {}}]
    rows = [{"claim_id": 1, "docs": ["b"]}, {"claim_id": 2, "docs": ["c"]}]
    metrics = hit_counts(rows, claims)
    assert metrics["all"] == metrics["pooling"] == {"hits": 1, "judged_claims": 1}
    assert metrics["citation"] == {"hits": 0, "judged_claims": 1}


def test_cli_abstains_or_returns_one_global_sentence(tmp_path, capsys):
    import json

    from evidence_retrieval.cli import main
    from evidence_retrieval.reader import Reader

    doc = {"doc_id": 4, "title": "Census", "abstract": ["Bird counts rose.", "The census concluded."]}
    (tmp_path / "corpus.jsonl").write_text(json.dumps(doc) + "\n")
    model = tmp_path / "reader.json"
    Reader([1., 0., 0., 0., 0., 0., 0.], 0., None).save(model)
    args = ["read", "Bird counts rose", "--data-dir", str(tmp_path), "--model", str(model)]
    assert main(args) == 0
    assert "Abstained" in capsys.readouterr().out
    assert main([*args, "--mode", "first"]) == 0
    assert capsys.readouterr().out.count("Sentence ") == 1
    assert main([*args, "--mode", "window", "--budget", "3"]) == 0
    assert capsys.readouterr().out.count("Sentence ") == 2
