"""Optional visualisation helpers.

Requires ``langchain-drift[viz]``:

    pip install langchain-drift[viz]

which installs ``matplotlib`` and ``scikit-learn``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from driftguard.corpus import ReferenceCorpus


def _tsne_2d(embeddings: np.ndarray, seed: int = 42) -> np.ndarray:
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        raise ImportError(
            "scikit-learn is required for visualisation. "
            "Install it with: pip install langchain-drift[viz]"
        )
    perplexity = min(30, max(2, len(embeddings) - 1))
    return TSNE(n_components=2, random_state=seed, perplexity=perplexity).fit_transform(embeddings)


def plot_corpus(
    corpus: "ReferenceCorpus",
    check_texts: list[str] | None = None,
    title: str | None = None,
    ax=None,
):
    """Plot the reference corpus and optionally overlay drift-checked texts.

    Reference texts are shown as blue circles.  When ``check_texts`` is
    provided, each text is embedded and checked; on-topic texts appear as
    green triangles, drift texts as red X markers.

    Uses t-SNE for dimensionality reduction (requires scikit-learn).

    Args:
        corpus: A ``ReferenceCorpus`` instance (must have at least 2 texts).
        check_texts: Texts to embed, check, and overlay on the plot.
        title: Plot title.
        ax: Existing ``matplotlib.axes.Axes`` to draw on.

    Returns:
        The ``matplotlib.figure.Figure`` object.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for visualisation. "
            "Install it with: pip install langchain-drift[viz]"
        )

    ref_embs = corpus.embeddings  # raises ValueError if empty

    check_vecs: np.ndarray | None = None
    is_drift_flags: list[bool] | None = None

    if check_texts:
        from driftguard.detector import DriftDetector
        detector = DriftDetector(corpus=corpus)
        check_vecs = np.array(corpus._model.embed_documents(list(check_texts)))
        is_drift_flags = [detector._evaluate(v, t, {}).is_drift
                          for v, t in zip(check_vecs, check_texts)]

    combined = np.vstack([ref_embs] + ([check_vecs] if check_vecs is not None else []))
    projected = _tsne_2d(combined)

    n_ref = len(ref_embs)
    ref_2d = projected[:n_ref]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    ax.scatter(ref_2d[:, 0], ref_2d[:, 1],
               c="steelblue", alpha=0.75, s=70, zorder=3,
               label=f"Reference ({n_ref})")
    for i, (x, y) in enumerate(ref_2d):
        label = corpus._texts[i][:28] if i < len(corpus._texts) else str(i)
        ax.annotate(label, (x, y), xytext=(4, 4), textcoords="offset points",
                    fontsize=7, alpha=0.6)

    if check_vecs is not None and is_drift_flags is not None:
        check_2d = projected[n_ref:]
        flags = np.array(is_drift_flags)
        ok_idx = np.where(~flags)[0]
        drift_idx = np.where(flags)[0]

        if len(ok_idx):
            ax.scatter(check_2d[ok_idx, 0], check_2d[ok_idx, 1],
                       c="seagreen", marker="^", s=90, alpha=0.85, zorder=4,
                       label=f"On-topic ({len(ok_idx)})")
        if len(drift_idx):
            ax.scatter(check_2d[drift_idx, 0], check_2d[drift_idx, 1],
                       c="crimson", marker="X", s=100, alpha=0.85, zorder=4,
                       label=f"Drift ({len(drift_idx)})")

        for i, (x, y) in enumerate(check_2d):
            color = "crimson" if is_drift_flags[i] else "seagreen"
            ax.annotate(check_texts[i][:28], (x, y), xytext=(4, 4),
                        textcoords="offset points", fontsize=7, color=color, alpha=0.75)

    ax.set_title(title or "Reference corpus (t-SNE)")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    return fig
