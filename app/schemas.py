from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class MatchupRequest(BaseModel):
    home_team: str
    away_team: str
    neutral_site: bool = False
    spread: float | None = None
    over_under: float | None = None
    kickoff: str | None = None
    venue: str | None = None
    season: int | None = None
    week: int | None = None
    game_id: int | None = None
    wx_temp_max: float | None = None
    wx_temp_min: float | None = None
    wx_precip: float | None = None
    wx_wind: float | None = None


class JobStartRequest(BaseModel):
    """Job launches take no parameters; unknown fields from older clients are ignored."""

    model_config = ConfigDict(extra="ignore")
