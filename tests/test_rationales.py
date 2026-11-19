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
