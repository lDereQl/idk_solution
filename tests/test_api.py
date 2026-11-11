from fastapi.testclient import TestClient

from app.main import app

c = TestClient(app)


def test_sites_and_models():
    assert len(c.get("/sites").json()) >= 1
    assert {m["id"] for m in c.get("/models").json()} >= {"lgbm_v1"}


def test_forecast_72_rows():
    r = c.get("/sites/flensburg/forecast", params={"model": "lgbm_v2"}).json()
    assert [x["horizon_h"] for x in r] == list(range(1, 73))
    assert r[0]["p10"] <= r[0]["p50"] <= r[0]["p90"]


def test_metrics_and_history():
    m = c.get("/sites/flensburg/metrics", params={"model": "lgbm_v2"}).json()
    assert 0 <= m["p10_p90_coverage"] <= 1
    h = c.get("/sites/flensburg/history", params={"start": "2020-01-01", "end": "2020-01-02"}).json()
    assert len(h) == 24


def test_404():
    assert c.get("/sites/nope/forecast", params={"model": "lgbm_v1"}).status_code == 404
    assert c.get("/sites/flensburg/forecast", params={"model": "nope"}).status_code == 404


def test_stub_data_valid():
    import subprocess
    import sys

    r = subprocess.run([sys.executable, "scripts/validate_data.py"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout


def test_point_baseline_forecast_has_null_band():
    r = c.get("/sites/kaunas_synthetic/forecast", params={"model": "baseline_24h"})
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) == 72 and rows[0]["p10"] is None and rows[0]["p50"] > 0


def test_history_gap_is_null(monkeypatch):
    import numpy as np

    from app import repository as repo

    h = repo.history("flensburg").copy()
    h.loc[h.index[-1], "heat_kwh"] = np.nan
    monkeypatch.setattr(repo, "history", lambda site_id: h)
    r = c.get("/sites/flensburg/history")
    assert r.status_code == 200 and r.json()[-1]["heat_kwh"] is None
