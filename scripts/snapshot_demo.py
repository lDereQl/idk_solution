"""Write web/in/kaunas_demo_data.json, the data the demo page inlines as a fallback when the API is not reachable.

    python scripts/snapshot_demo.py [data dir, default demo_data]

Reads the data through the API itself, so the page gets exactly what the live API would return. Run it after
export_demo_data.py, then python web/build.py.
"""
import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
os.environ["DATA_DIR"] = str(Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "demo_data").resolve())
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

SITE = "kaunas_synthetic"  # the page's SITE_ID
SHOWN_ISSUES = 7           # the page shows the last 7 issues
c = TestClient(app)


def get(path, **params):
    r = c.get(path, params=params)
    r.raise_for_status()
    return r.json()


models = get("/models")
issues = get(f"/sites/{SITE}/issues", model=models[-1]["id"])
shown = issues[-SHOWN_ISSUES:]
start = (pd.Timestamp(shown[0]) - pd.Timedelta(hours=168)).strftime("%Y-%m-%dT%H:%M:%SZ")
end = (pd.Timestamp(shown[-1]) + pd.Timedelta(hours=73)).strftime("%Y-%m-%dT%H:%M:%SZ")
data = {
    "site": next(s for s in get("/sites") if s["site_id"] == SITE),
    "models": models,
    "issues": issues,
    "forecasts": {m["id"]: {i: get(f"/sites/{SITE}/forecast", model=m["id"], issue=i) for i in shown} for m in models},
    "metrics": {m["id"]: get(f"/sites/{SITE}/metrics", model=m["id"]) for m in models},
    "history": get(f"/sites/{SITE}/history", start=start, end=end),
}
path = ROOT / "web" / "in" / "kaunas_demo_data.json"
path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
print(f"wrote {path.relative_to(ROOT)}: {len(models)} models, {len(issues)} issues, {len(data['history'])} history hours")
