from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_train_page_and_client_have_no_holdout_field():
    page = (ROOT / "web/src/pages/Train.jsx").read_text()
    api = (ROOT / "web/src/api.js").read_text()
    pipeline = (ROOT / "run_pipeline.py").read_text()
    schema = (ROOT / "app/schemas.py").read_text()
    for text in (page, api, pipeline, schema):
        assert "holdout" not in text.lower()
