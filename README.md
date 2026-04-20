# driftguard

Embedding-based response drift detection for LangChain agents.

Detects when an LLM starts answering outside its intended domain (a legal assistant drifting into cooking advice, a medical chatbot wandering into finance) without ground-truth labels or a separate classifier.

---

- [How it works](#how-it-works)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Integration patterns](#integration-patterns)
  - [Callback (non-blocking)](#pattern-1--callback-non-blocking-observer)
  - [Runnable guard (blocking)](#pattern-2--runnable-guard-blocking)
  - [Passthrough (annotate only)](#pattern-3--passthrough-annotate-dont-block)
- [LangGraph guardrail](#langgraph-guardrail)
- [Async support](#async-support)
- [Alert sinks](#alert-sinks)
- [Multi-topic corpora](#multi-topic-corpora-clustering)
- [Domain auditing](#domain-auditing)
- [Building a corpus with FPS](#building-a-corpus-with-fps)
- [Distribution-level detection](#distribution-level-detection-windowed)
- [Visualisation](#visualisation)
- [Persisting a corpus](#persisting-a-corpus)
- [DriftResult reference](#driftresult-reference)
- [Development](#development)

---

## How it works

1. **Build a reference corpus** from representative on-topic texts.
2. **Embed each LLM response** with the same model.
3. **Compare** using two complementary signals:
   - *Centroid distance*: how close is the response to the centre of the corpus (or its nearest cluster)?
   - *Nearest-neighbour distance*: is the response close to at least one reference text?
4. **Flag drift** when both signals agree the response is far from the reference domain.

Using both signals reduces false positives: a paraphrase that sits slightly off the centroid is rescued when it's still close to a known reference text.

The threshold for each signal is **adaptive**: the 5th percentile of within-corpus similarity scores, so ~95% of reference texts clear it with no manual tuning.

---

## Installation

```bash
git clone https://github.com/moudather/driftguard.git
cd driftguard
pip install -r requirements.txt
pip install -e .
```

Requires Python ≥ 3.9. The only runtime dependencies are `langchain-core` and `numpy`.

Optional extras:

```bash
pip install -e ".[viz]"        # matplotlib + scikit-learn for corpus.plot()
pip install langgraph          # LangGraph guardrail nodes
```

---

## Quick start

```python
from driftguard import ReferenceCorpus, DriftDetector
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

# 1. Build the reference corpus from representative on-topic texts
corpus = ReferenceCorpus(embeddings_model=embeddings)
corpus.add_texts([
    "tort law", "contract formation", "negligence standard",
    "criminal intent", "due process rights",
])

# 2. Create the detector
detector = DriftDetector(corpus=corpus)

# 3. Check a response
result = detector.check("habeas corpus")
print(result.is_drift)                # False (on-topic)
print(result.centroid_similarity)     # e.g. 0.91
print(result.max_reference_similarity)# e.g. 0.95
print(result.threshold)               # e.g. 0.87

result = detector.check("best pasta recipe")
print(result.is_drift)                # True (off-topic)
```

---

## Integration patterns

### Pattern 1: Callback (non-blocking observer)

Attach to any LangChain LLM or chat model. Runs on every response without interrupting the pipeline; use for monitoring, logging, or metrics.

```python
from driftguard import DriftCallbackHandler, AlertManager

alerts = AlertManager(sinks=["log"])
handler = DriftCallbackHandler(detector=detector, alerts=alerts)

llm = ChatOpenAI(callbacks=[handler])
response = llm.invoke("What is the recipe for tiramisu?")
# Drift is logged as a WARNING; the response still returns normally.

print(handler.history[-1].is_drift)   # True
```

### Pattern 2: Runnable guard (blocking)

Insert as a step in a LangChain chain. Raises `DriftError` on drift; passes the text through unchanged otherwise.

```python
from driftguard import DriftRunnable, DriftError
from langchain_core.output_parsers import StrOutputParser

drift = DriftRunnable(detector=detector)
chain = llm | StrOutputParser() | drift.as_guard()

try:
    result = chain.invoke("What is the recipe for tiramisu?")
except DriftError as e:
    print(f"Blocked: centroid_sim={e.result.centroid_similarity:.3f} "
          f"< threshold={e.result.threshold:.3f}")
```

### Pattern 3: Passthrough (annotate, don't block)

Annotates the chain output with drift metadata without halting. Useful when you want to observe drift but let the response through for the user to see.

```python
chain = llm | StrOutputParser() | drift.as_passthrough()
output = chain.invoke("habeas corpus")
# {"output": "Habeas corpus is a legal right...", "drift": DriftResult(...)}
print(output["drift"].is_drift)       # False
```

---

## LangGraph guardrail

`driftguard` ships a first-class LangGraph integration. The node and routing helpers are plain callables that match LangGraph's expected signatures, no LangGraph import inside the library itself, so the module loads fine even if LangGraph isn't installed.

```python
from langgraph.graph import StateGraph
from typing import Any
from typing_extensions import TypedDict

from driftguard.langgraph import drift_node, route_on_drift

class AgentState(TypedDict):
    query: str
    response: str
    drift: Any          # holds the DriftResult written by the drift node

graph = StateGraph(AgentState)

graph.add_node("llm", call_llm)                     # writes state["response"]
graph.add_node("drift_check", drift_node(detector)) # reads "response", writes "drift"
graph.add_node("fallback", handle_fallback)
graph.add_node("respond", finalize)

graph.set_entry_point("llm")
graph.add_edge("llm", "drift_check")
graph.add_conditional_edges(
    "drift_check",
    route_on_drift,                 # returns "drift" or "ok"
    {"drift": "fallback", "ok": "respond"},
)

app = graph.compile()
```

**Custom state key**: if your LLM node writes to a key other than `"response"`:

```python
graph.add_node("drift_check", drift_node(detector, text_key="output"))
```

**Async graphs**: swap `drift_node` for `adrift_node`:

```python
from driftguard.langgraph import adrift_node

graph.add_node("drift_check", adrift_node(detector))
```

**Custom route labels**: use `make_route_on_drift` when your edge map uses different names:

```python
from driftguard.langgraph import make_route_on_drift

router = make_route_on_drift(on_drift="blocked", on_ok="continue")
graph.add_conditional_edges(
    "drift_check", router, {"blocked": "fallback", "continue": "respond"}
)
```

---

## Async support

Every public method has an async counterpart:

```python
await corpus.aadd_texts(["tort law", "negligence"])
result = await detector.acheck("contract formation")
```

`AsyncDriftCallbackHandler` mirrors `DriftCallbackHandler` for async LangChain pipelines.

---

## Alert sinks

`AlertManager` dispatches drift alerts to one or more sinks simultaneously:

```python
from driftguard import AlertManager

alerts = AlertManager(sinks=[
    "log",                                       # WARNING via Python logging
    "https://your-service.example/webhook",      # POST JSON payload
    lambda result: my_queue.put(result),         # arbitrary sync or async callable
])
```

Pass an `AlertManager` instance to `DriftCallbackHandler`, `DriftRunnable`, or the LangGraph nodes; all accept one via the `alerts` argument.

---

## Multi-topic corpora (clustering)

When your reference corpus spans several distinct topics, a single global centroid produces false positives for texts that are on-topic but far from the average. Set `n_clusters` to partition the corpus into groups; each query is then compared to its nearest cluster rather than the global centre.

```python
corpus = ReferenceCorpus(embeddings_model=embeddings, n_clusters=2)
corpus.add_texts([
    # Legal cluster
    "tort law", "contract formation", "negligence",
    # Medical cluster
    "malpractice", "diagnosis", "clinical trial",
])

detector = DriftDetector(corpus=corpus)

detector.check("habeas corpus").is_drift   # False (routes to legal cluster)
detector.check("prognosis").is_drift       # False (routes to medical cluster)
detector.check("pasta recipe").is_drift    # True  (far from both clusters)
```

Clustering uses numpy k-means internally with no extra dependencies.

---

## Domain auditing

The `Auditor` class runs drift detection over a batch of historical responses and returns a structured report: pass rate, score distribution, flagged outliers. Use it before deployment to validate your corpus, after incidents to understand what went wrong, or in CI to catch domain regressions between prompt versions.

```python
from driftguard import Auditor

auditor = Auditor(detector)
report = auditor.run(production_responses)

print(f"Pass rate:  {report.pass_rate:.1%}")
print(f"Drift rate: {report.drift_rate:.1%}")
print(f"Flagged:    {report.flagged} / {report.total}")
```

**Export the report** for a compliance doc or CI artifact:

```python
report.to_json()             # structured JSON string
open("report.html", "w").write(report.to_html())  # self-contained HTML report
```

The HTML report includes a summary dashboard, centroid similarity distribution (p5 → p95), and a table of all flagged responses with their scores.

**Async**: all responses are checked concurrently:

```python
report = await auditor.arun(production_responses)
```

### Comparing corpora

Detect domain shift between prompt versions, model upgrades, or dataset changes:

```python
comparison = corpus_v1.compare(corpus_v2)

print(f"Centroid shift: {comparison.centroid_shift:.4f}")  # cosine distance
print(f"Threshold delta: {comparison.threshold_delta:+.4f}")
print(f"Significant: {comparison.is_significant}")         # shift > 0.05
```

A `centroid_shift` above 0.05 (configurable via `significant_shift_threshold`) means the two corpora represent meaningfully different domains, worth investigating before swapping one for the other.

---

## Building a corpus with FPS

Hand-picking reference texts is tedious and easy to get wrong. `ReferenceCorpus.from_texts()` accepts a large pool of candidates and uses **Farthest Point Sampling** to automatically select the `n` most coverage-maximising texts; each new selection is the one farthest (in cosine distance) from all already-chosen texts.

```python
# 500 example legal responses; pick the 30 most diverse ones.
corpus = ReferenceCorpus.from_texts(
    candidates=my_500_legal_responses,
    embeddings_model=embeddings,
    n=30,
)
```

The result is a fully initialised `ReferenceCorpus` ready for use with `DriftDetector`. An async variant is also available:

```python
corpus = await ReferenceCorpus.afrom_texts(candidates, embeddings_model=embeddings, n=30)
```

---

## Distribution-level detection (windowed)

Per-response checks are sensitive to one-off anomalies. `WindowedDriftDetector` accumulates a sliding window of responses and checks whether the *window's* embedding distribution has shifted from the reference. Two signals can trigger drift:

- **Centroid shift**: the window's mean embedding has moved away from the reference.
- **Drift fraction**: more than `drift_fraction_threshold` (default 30%) of recent responses are individually off-topic.

```python
from driftguard import WindowedDriftDetector

wd = WindowedDriftDetector(corpus=corpus, window_size=20, drift_fraction_threshold=0.3)

for response in llm_responses:
    result = wd.update(response)
    if result is None:
        continue   # window still filling
    if result.is_drift:
        print(f"Window drift detected: "
              f"centroid_sim={result.window_centroid_similarity:.3f}, "
              f"drift_fraction={result.drift_fraction:.0%}")
```

`result` is a `WindowDriftResult` returned on every call once the window is full. Use `on_drift` for async-friendly callbacks:

```python
wd = WindowedDriftDetector(corpus=corpus, on_drift=lambda r: alert_queue.put(r))
```

Async usage mirrors the sync API:

```python
result = await wd.aupdate(response)
```

---

## Visualisation

`corpus.plot()` projects the reference corpus into 2D via t-SNE and optionally overlays texts colour-coded by drift status, useful for debugging false positives and tuning `threshold_percentile`.

```bash
pip install driftguard[viz]   # adds matplotlib + scikit-learn
```

```python
corpus.plot(check_texts=["habeas corpus", "pasta recipe", "clinical trial"])
```

Blue circles are reference texts; green triangles are on-topic detections; red X markers are flagged as drift.

For more control, call `plot_corpus` directly:

```python
from driftguard.viz import plot_corpus
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 7))
plot_corpus(corpus, check_texts=probe_texts, ax=ax)
plt.show()
```

---

## Persisting a corpus

Save a trained corpus to disk and reload it on the next run, no need to re-embed reference texts every time.

```python
corpus.save("legal_corpus")
# writes legal_corpus.npz  (embeddings, centroid, thresholds, cluster data)
#        legal_corpus.texts.json  (original texts)

loaded = ReferenceCorpus(embeddings_model=embeddings)
loaded.load("legal_corpus")
```

Cluster data (centroids, per-cluster thresholds) is persisted alongside the embeddings.

---

## DriftResult reference

Every call to `detector.check()` or `detector.acheck()` returns a frozen `DriftResult`:

| Field | Type | Description |
|-------|------|-------------|
| `is_drift` | `bool` | `True` when both centroid and NN signals indicate drift |
| `centroid_similarity` | `float` | Cosine similarity to the nearest cluster (or global) centroid |
| `max_reference_similarity` | `float` | Cosine similarity to the closest individual reference text |
| `threshold` | `float` | Adaptive centroid threshold for this check |
| `nn_threshold` | `float` | Adaptive nearest-neighbour threshold |
| `text` | `str` | The checked text |
| `timestamp` | `float` | Unix timestamp |
| `metadata` | `dict` | Any kwargs passed to `check()`, e.g. `run_id` |

`DriftError` (raised by `as_guard()`) exposes the full `DriftResult` on its `.result` attribute.

### WindowDriftResult

`WindowedDriftDetector.update()` returns a `WindowDriftResult` once the window is full:

| Field | Type | Description |
|-------|------|-------------|
| `is_drift` | `bool` | `True` when centroid or fraction signal fires |
| `window_centroid_similarity` | `float` | Cosine similarity of window centroid to reference |
| `drift_fraction` | `float` | Fraction of window responses individually flagged |
| `window_size` | `int` | Number of responses in the window |
| `threshold` | `float` | Reference threshold used for centroid check |
| `drift_fraction_threshold` | `float` | Configured fraction threshold |
| `timestamp` | `float` | Unix timestamp |

---

## Development

```bash
pip install -e ".[dev]"
pytest
```

All tests use deterministic `FakeEmbeddings`, no API key or network access required.

---

## License

MIT
