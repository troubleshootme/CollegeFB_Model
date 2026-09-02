"""Ingest package. Import flatten without requiring the CFBD SDK."""

from __future__ import annotations

__all__ = ["run_ingest"]


def __getattr__(name: str):
    if name == "run_ingest":
        from cfb_model.ingest.pipeline import run_ingest

        return run_ingest
    raise AttributeError(f"module {__name__!r} has no attribute {name}")
