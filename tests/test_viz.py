"""Visualisation tests — skipped automatically if matplotlib is not installed."""
import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from driftguard.corpus import ReferenceCorpus
from driftguard.viz import plot_corpus
from tests.conftest import FakeEmbeddings


def _make_corpus():
    legal_base = np.array([1.0, 0, 0, 0, 0, 0, 0, 0])
    food_base = np.array([0, 0, 0, 0, 1.0, 0, 0, 0])
    mapping = {
        "tort law": (legal_base + np.array([0.05, 0, 0, 0, 0, 0, 0, 0])).tolist(),
        "contract law": (legal_base + np.array([0.10, 0, 0, 0, 0, 0, 0, 0])).tolist(),
        "negligence": (legal_base + np.array([0, 0.05, 0, 0, 0, 0, 0, 0])).tolist(),
        "habeas corpus": (legal_base + np.array([0.07, 0, 0, 0, 0, 0, 0, 0])).tolist(),
        "pasta recipe": food_base.tolist(),
    }
    emb = FakeEmbeddings(mapping=mapping)
    corpus = ReferenceCorpus(embeddings_model=emb)
    corpus.add_texts(["tort law", "contract law", "negligence"])
    return corpus


def test_plot_corpus_returns_figure():
    import matplotlib.pyplot as plt
    fig = plot_corpus(_make_corpus())
    assert fig is not None
    plt.close("all")


def test_plot_corpus_with_check_texts():
    import matplotlib.pyplot as plt
    fig = plot_corpus(_make_corpus(), check_texts=["habeas corpus", "pasta recipe"])
    assert fig is not None
    plt.close("all")


def test_corpus_plot_convenience_method():
    import matplotlib.pyplot as plt
    fig = _make_corpus().plot()
    assert fig is not None
    plt.close("all")


def test_plot_corpus_empty_raises():
    emb = FakeEmbeddings()
    empty = ReferenceCorpus(embeddings_model=emb)
    with pytest.raises(ValueError, match="empty"):
        plot_corpus(empty)
