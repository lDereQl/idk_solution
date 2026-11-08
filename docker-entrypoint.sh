#!/bin/sh
set -e
# No real data mounted -> generate synthetic stubs so the API still starts.
if [ ! -f "$DATA_DIR/sites.csv" ]; then
  python scripts/make_stub_data.py
fi
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
