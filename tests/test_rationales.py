from evidence_retrieval.eval.rationales import EvidenceRetriever, score_claim


def test_two_stage_retrieval_and_multi_sentence_credit():
    docs = [
        {"doc_id": 1, "title": "Ocean survey", "abstract": ["Methods varied.", "Ocean temperatures rose."]},
        {"doc_id": 2, "title": "Bird census", "abstract": ["Sparrows nested in barns."]},
    ]
    retriever = EvidenceRetriever(docs)
    hits = retriever.retrieve("Ocean temperatures rose", k=1)
    claim = {"id": 1, "claim": "Ocean temperatures rose",
             "evidence": {"1": [{"label": "SUPPORT", "sentences": [0, 1]}]}}
    row = score_claim(claim, hits)
    assert row["abstract_hit"] and row["sentence_hit"] and not row["complete_hit"]
    assert row["matches"][0]["sentence_id"] == 1
    claim["evidence"]["1"].append({"label": "SUPPORT", "sentences": [1]})
    assert score_claim(claim, hits)["complete_hit"]
    assert not score_claim(claim, retriever.retrieve("sparrows nested", k=1))["abstract_hit"]


def test_retrieved_abstract_can_miss_the_rationale():
    claim = {"id": 1, "claim": "Cells divided", "evidence": {
        "2": [{"label": "CONTRADICT", "sentences": [0]}]}}
    result = score_claim(claim, [{"doc_id": "2", "sentence_id": 1}])
    assert result["abstract_hit"] and not result["sentence_hit"]


def test_evidence_cli_both_selectors(tmp_path, capsys):
    import json

    from evidence_retrieval.cli import main

    docs = [{"doc_id": 1, "title": "Bird census",
             "abstract": ["Observers recorded birds.", "RESULTS Sparrow numbers rose."]}]
    train = [{"id": 2, "claim": "Sparrow numbers rose.",
              "evidence": {"1": [{"label": "SUPPORT", "sentences": [1]}]}}]
    (tmp_path / "corpus.jsonl").write_text(json.dumps(docs[0]) + "\n")
    args = ["evidence", "Sparrow numbers rose", "--data-dir", str(tmp_path), "--top-k", "1"]
    assert main(args) == 0
    assert "Sentence 2: RESULTS Sparrow numbers rose." in capsys.readouterr().out
    (tmp_path / "claims_train.jsonl").write_text(json.dumps(train[0]) + "\n")
    assert main([*args, "--selector", "learned"]) == 0
    result = capsys.readouterr().out
    assert "[SciFact 1]" in result and "Sentence 2:" in result
