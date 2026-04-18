import logging

from driftguard.alerts import AlertManager
from driftguard.schema import DriftResult


def _make_result(is_drift=True):
    return DriftResult(
        is_drift=is_drift,
        centroid_similarity=0.3,
        max_reference_similarity=0.4,
        threshold=0.8,
        nn_threshold=0.75,
        text="some off-topic text",
    )


def test_log_sink(caplog):
    alerts = AlertManager(sinks=["log"])
    result = _make_result()
    with caplog.at_level(logging.WARNING, logger="driftguard"):
        alerts.alert(result)
    assert "DRIFT DETECTED" in caplog.text
    assert "0.3000" in caplog.text


def test_callable_sink():
    received = []
    alerts = AlertManager(sinks=[lambda r: received.append(r)])
    result = _make_result()
    alerts.alert(result)
    assert len(received) == 1
    assert received[0] is result


def test_multiple_sinks(caplog):
    received = []
    alerts = AlertManager(sinks=["log", lambda r: received.append(r)])
    result = _make_result()
    with caplog.at_level(logging.WARNING, logger="driftguard"):
        alerts.alert(result)
    assert "DRIFT DETECTED" in caplog.text
    assert len(received) == 1
