from cfb_model.identity import expected_pass_yards, expected_rush_yards
from cfb_model.quota import live_seasons, years_to_ingest


def test_rushing_yards_move_more_with_the_opponent_than_passing():
    rush = expected_rush_yards(150, 80)
    pas = expected_pass_yards(250, 180)
    assert abs(150 - rush) > abs(250 - pas)


def test_years_to_ingest_skips_stored_history_but_refreshes_live(monkeypatch):
    monkeypatch.setattr("cfb_model.quota.live_seasons", lambda today=None: {2026})
    years = years_to_ingest(2020, 2026, existing={2020, 2021, 2022, 2023, 2024, 2025}, force=False)
    assert years == [2026]
    forced = years_to_ingest(2024, 2026, existing={2024, 2025, 2026}, force=True)
    assert forced == [2024, 2025, 2026]
    assert 2026 in live_seasons()
