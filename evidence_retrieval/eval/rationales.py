"""Two-stage SciFact evidence retrieval using the original sentence boundaries."""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from evidence_retrieval.encoders import BM25Encoder


class EvidenceRetriever:
    """Return one sentence per candidate abstract, without consulting judgments."""

    def __init__(self, corpus):
        self.documents = list(corpus)
        self.offsets = [0]
        sentences = []
        passages = []
        for doc in self.documents:
            sentences.extend(doc["abstract"])
            self.offsets.append(len(sentences))
            passages.append(" ".join([doc["title"], *doc["abstract"]]))
        self.bm25 = BM25Encoder().fit(passages)
        self.vectorizer = TfidfVectorizer(sublinear_tf=True)
        self.vectors = self.vectorizer.fit_transform(sentences)

    def retrieve(self, claim, k=10):
        query = self.vectorizer.transform([claim])
        hits = []
        for index, _ in self.bm25.rank(claim, k):
            start, end = self.offsets[index:index + 2]
            # Abstract order breaks ties, including queries outside the vocabulary.
            position = None
            if end > start:
                scores = self.vectors[start:end] @ query.T
                position = int(np.argmax(scores.toarray()))
            doc = self.documents[index]
            hits.append({"doc_id": str(doc["doc_id"]), "title": doc["title"],
                         "sentence_id": position,
                         "sentence": doc["abstract"][position] if position is not None else ""})
        return hits


def score_claim(claim, hits):
    """Credit a claim once, even if multiple abstracts or rationales match."""
    gold = claim["evidence"]
    matches = []
    for rank, hit in enumerate(hits, 1):
        groups = gold.get(hit["doc_id"], [])
        if groups:
            selected = hit["sentence_id"]
            matches.append({
                "rank": rank, **hit, "rationales": groups,
                "sentence_hit": any(selected in g["sentences"] for g in groups),
                "complete_hit": any(set(g["sentences"]) == {selected} for g in groups),
            })
    return {
        "claim_id": claim["id"], "claim": claim["claim"],
        "abstract_hit": bool(matches),
        "sentence_hit": any(m["sentence_hit"] for m in matches),
        "complete_hit": any(m["complete_hit"] for m in matches),
        "lead_hit": any(0 in g["sentences"] for m in matches for g in m["rationales"]),
        "gold": gold, "matches": matches, "candidates": hits,
    }
