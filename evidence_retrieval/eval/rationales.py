"""Two-stage SciFact evidence retrieval using the original sentence boundaries."""

import json
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from evidence_retrieval.encoders import BM25Encoder


def records(path):
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def sentence_features(sentences, similarity):
    """Numeric inputs available at inference, in original sentence order."""
    n = len(sentences)
    return np.column_stack([
        similarity,
        similarity / max(similarity.max(), 1e-9),
        np.arange(n) / max(n - 1, 1),
        [bool(re.match(r"^\s*(?:results?|conclusions?|findings)\b", s, re.I)) for s in sentences],
        [bool(re.search(r"\d", s)) for s in sentences],
        np.log1p([len(s.split()) for s in sentences]),
    ])


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

    def _similarity(self, index, query):
        start, end = self.offsets[index:index + 2]
        return (self.vectors[start:end] @ query.T).toarray().ravel()

    def fit_selector(self, claims):
        """Learn sentence relevance inside annotated training abstracts."""
        indices = {str(doc["doc_id"]): i for i, doc in enumerate(self.documents)}
        features, targets = [], []
        for claim in claims:
            query = self.vectorizer.transform([claim["claim"]])
            for doc_id, rationales in claim["evidence"].items():
                index = indices[doc_id]
                sentences = self.documents[index]["abstract"]
                positive = {i for group in rationales for i in group["sentences"]}
                features.append(sentence_features(sentences, self._similarity(index, query)))
                targets.extend(i in positive for i in range(len(sentences)))
        return LogisticRegression(max_iter=500).fit(np.vstack(features), targets)

    def retrieve(self, claim, k=10, selector=None):
        query = self.vectorizer.transform([claim])
        hits = []
        for index, _ in self.bm25.rank(claim, k):
            doc = self.documents[index]
            sentences = doc["abstract"]
            # Abstract order breaks ties, including queries outside the vocabulary.
            position = None
            if sentences:
                scores = self._similarity(index, query)
                if selector is not None:
                    scores = selector.decision_function(sentence_features(sentences, scores))
                position = int(np.argmax(scores))
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
