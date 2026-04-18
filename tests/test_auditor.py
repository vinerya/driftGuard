import json

import numpy as np
import pytest

from driftguard import (
    Auditor,
    AuditReport,
    CorpusComparison,
    DriftDetector,
    ReferenceCorpus,
)
from tests.conftest import FakeEmbeddings


def _make_system():
    legal_base = np.array([1.0, 0, 0, 0, 0, 0, 0, 0])
    food_base = np.array([0, 0, 0, 0, 1.0, 0, 0, 0])
    mapping = {
        "tort law": (legal_base + np.array([0.05, 0, 0, 0, 0, 0, 0, 0])).tolist(),
        "contract law": (legal_base + np.array([0.10, 0, 0, 0, 0, 0, 0, 0])).tolist(),
        "negligence": (legal_base + np.array([0, 0.05, 0, 0, 0, 0, 0, 0])).tolist(),
        "due process": (legal_base + np.array([0.08, 0.03, 0, 0, 0, 0, 0, 0])).tolist(),
        "habeas corpus": (legal_base + np.array([0.07, 0, 0, 0, 0, 0, 0, 0])).tolist(),
        "summary judgment": (legal_base + np.array([0.06, 0.02, 0, 0, 0, 0, 0, 0])).tolist(),
        "pasta recipe": food_base.tolist(),
        "weather forecast": np.array([0, 0, 0, 0, 0, 1.0, 0, 0]).tolist(),
    }
    emb = FakeEmbeddings(mapping=mapping)
    corpus = ReferenceCorpus(embeddings_model=emb)
    corpus.add_texts(["tort law", "contract law", "negligence", "due process"])
    detector = DriftDetector(corpus=corpus)
    return emb, corpus, detector


# --- Auditor.run ---

def test_auditor_run_returns_report():
    _, _, detector = _make_system()
    auditor = Auditor(detector)
    report = auditor.run(["habeas corpus", "summary judgment"])
    assert isinstance(report, AuditReport)


def test_auditor_pass_rate_all_on_topic():
    _, _, detector = _make_system()
    auditor = Auditor(detector)
    report = auditor.run(["habeas corpus", "summary judgment", "due process"])
    assert report.pass_rate == 1.0
    assert report.flagged == 0


def test_auditor_drift_rate_all_off_topic():
    _, _, detector = _make_system()
    auditor = Auditor(detector)
    report = auditor.run(["pasta recipe", "weather forecast"])
    assert report.drift_rate == 1.0
    assert report.passed == 0


def test_auditor_mixed_responses():
    _, _, detector = _make_system()
    auditor = Auditor(detector)
    report = auditor.run([
        "habeas corpus",    # on-topic
        "summary judgment", # on-topic
        "pasta recipe",     # drift
        "weather forecast", # drift
    ])
    assert report.total == 4
    assert report.passed == 2
    assert report.flagged == 2
    assert abs(report.pass_rate - 0.5) < 1e-9
    assert abs(report.drift_rate - 0.5) < 1e-9


def test_auditor_outliers_are_drift_results():
    _, _, detector = _make_system()
    auditor = Auditor(detector)
    report = auditor.run(["habeas corpus", "pasta recipe"])
    assert all(r.is_drift for r in report.outliers)
    assert len(report.outliers) == report.flagged


def test_auditor_distribution_fields():
    _, _, detector = _make_system()
    auditor = Auditor(detector)
    report = auditor.run(["habeas corpus", "summary judgment", "pasta recipe"])
    assert report.centroid_similarity_p50 >= report.centroid_similarity_p5
    assert report.centroid_similarity_p95 >= report.centroid_similarity_p50


def test_auditor_empty_raises():
    _, _, detector = _make_system()
    with pytest.raises(ValueError, match="empty"):
        Auditor(detector).run([])


async def test_auditor_arun():
    _, _, detector = _make_system()
    auditor = Auditor(detector)
    report = await auditor.arun(["habeas corpus", "pasta recipe"])
    assert report.total == 2
    assert report.flagged == 1


# --- AuditReport serialisation ---

def test_to_dict_structure():
    _, _, detector = _make_system()
    report = Auditor(detector).run(["habeas corpus", "pasta recipe"])
    d = report.to_dict()
    assert "summary" in d
    assert "distribution" in d
    assert "outliers" in d
    assert d["summary"]["total"] == 2


def test_to_json_valid():
    _, _, detector = _make_system()
    report = Auditor(detector).run(["habeas corpus", "pasta recipe"])
    parsed = json.loads(report.to_json())
    assert parsed["summary"]["total"] == 2


def test_to_html_contains_key_fields():
    _, _, detector = _make_system()
    report = Auditor(detector).run(["habeas corpus", "pasta recipe"])
    html = report.to_html()
    assert "Drift Audit Report" in html
    assert "Pass rate" in html
    assert "pasta recipe" in html


# --- CorpusComparison ---

def test_compare_identical_corpora():
    _, corpus, _ = _make_system()
    comparison = corpus.compare(corpus)
    assert comparison.centroid_shift < 1e-9
    assert not comparison.is_significant


def test_compare_different_corpora():
    legal_base = np.array([1.0, 0, 0, 0, 0, 0, 0, 0])
    medical_base = np.array([0, 1.0, 0, 0, 0, 0, 0, 0])
    mapping = {
        "tort law": (legal_base + np.array([0.05, 0, 0, 0, 0, 0, 0, 0])).tolist(),
        "contract law": (legal_base + np.array([0.10, 0, 0, 0, 0, 0, 0, 0])).tolist(),
        "negligence": (legal_base + np.array([0, 0.05, 0, 0, 0, 0, 0, 0])).tolist(),
        "malpractice": (medical_base + np.array([0, 0.05, 0, 0, 0, 0, 0, 0])).tolist(),
        "diagnosis": (medical_base + np.array([0, 0.10, 0, 0, 0, 0, 0, 0])).tolist(),
        "prognosis": (medical_base + np.array([0, 0, 0.05, 0, 0, 0, 0, 0])).tolist(),
    }
    emb = FakeEmbeddings(mapping=mapping)

    corpus_v1 = ReferenceCorpus(embeddings_model=emb)
    corpus_v1.add_texts(["tort law", "contract law", "negligence"])

    corpus_v2 = ReferenceCorpus(embeddings_model=emb)
    corpus_v2.add_texts(["malpractice", "diagnosis", "prognosis"])

    comparison = corpus_v1.compare(corpus_v2)
    assert comparison.centroid_shift > 0.5
    assert comparison.is_significant
    assert comparison.size_delta == 0


def test_compare_to_dict():
    _, corpus, _ = _make_system()
    d = corpus.compare(corpus).to_dict()
    assert "centroid_shift" in d
    assert "is_significant" in d
