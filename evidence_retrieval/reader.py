"""Select evidence across BM25 candidates at a fixed reading budget."""

from dataclasses import asdict, dataclass
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit

from evidence_retrieval.eval.rationales import sentence_features


@dataclass
class Candidate:
    doc: dict
    features: np.ndarray


def candidates(retriever, claim):
    query = retriever.vectorizer.transform([claim])
    result = []
    for rank, (idx, _) in enumerate(retriever.bm25.rank(claim, 10), start=1):
        doc = retriever.documents[idx]
        if doc["abstract"]:
            sim = retriever._similarity(idx, query)
            features = sentence_features(doc["abstract"], sim)
            features = np.column_stack((features, np.full(len(sim), 1 / rank)))
            result.append(Candidate(doc, features))
    return result


def training_split(claims):
    """Keep claims connected by any cited/evidence paper on the same side."""
    parents = {}

    def root(doc):
        parents.setdefault(doc, doc)
        while parents[doc] != doc:
            parents[doc] = parents[parents[doc]]
            doc = parents[doc]
        return doc

    groups = []
    for claim in claims:
        docs = sorted({str(i) for i in claim["cited_doc_ids"]} | set(claim["evidence"]))
        for doc in docs[1:]:
            parents[root(doc)] = root(docs[0])
        groups.append(docs[0])
    split = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    fit, calibration = next(split.split(claims, groups=[root(doc) for doc in groups]))
    return [claims[i] for i in fit], [claims[i] for i in calibration]


@dataclass
class Reader:
    weights: list
    intercept: float
    threshold: float | None = None

    @classmethod
    def load(cls, path):
        return cls(**json.loads(path.read_text()))

    def save(self, path):
        path.write_text(json.dumps(asdict(self), indent=2, allow_nan=False) + "\n")

    @classmethod
    def fit(cls, retriever, claims):
        features, labels = [], []
        for claim in claims:
            for candidate in candidates(retriever, claim["claim"]):
                groups = claim["evidence"].get(str(candidate.doc["doc_id"]), [])
                positive = {i for g in groups for i in g["sentences"]}
                features.append(candidate.features)
                labels.extend(i in positive for i in range(len(candidate.features)))
        model = LogisticRegression(max_iter=500).fit(np.vstack(features), labels)
        return cls(model.coef_[0].tolist(), float(model.intercept_[0]))

    def score(self, candidates):
        return [(c.doc, c.features @ self.weights + self.intercept) for c in candidates]

    def accepts(self, score):
        return self.threshold is not None and score >= self.threshold


def first_sentence(scored):
    """Top sentence globally. Ties preserve BM25 rank and sentence order."""
    best = None
    for doc, scores in scored:
        i = int(np.argmax(scores))
        if best is None or scores[i] > best[2]:
            best = (doc, i, float(scores[i]))
    return best


def select_budget(scores, budget, adjacent):
    width = min(budget, len(scores))
    if adjacent:
        start = int(np.argmax(np.convolve(scores, np.ones(width), mode="valid")))
        return list(range(start, start + width))
    return sorted(np.argsort(-scores, kind="stable")[:width].tolist())


def is_evidence(claim, doc, index):
    return any(index in group["sentences"] for group in claim["evidence"].get(str(doc["doc_id"]), []))


def calibrate(observations, target=0.70, minimum=10):
    """Most covered observed cutoff meeting the target, with all ties included."""
    ordered = sorted(observations, reverse=True)
    threshold = None
    correct = 0
    for count, (score, hit) in enumerate(ordered, 1):
        correct += hit
        if count < len(ordered) and ordered[count][0] == score:
            continue
        if count >= minimum and correct >= target * count:
            threshold = score
    return threshold
