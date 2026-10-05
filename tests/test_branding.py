from src.collect import branding_fields, fetch_fbs_teams


def test_branding_fields_normalize_hex_and_prefer_logos_list():
    team = {
        "id": 333,
        "school": "Alabama",
        "abbreviation": "ALA",
        "conference": "SEC",
        "color": "9e1b32",
        "altColor": "ffffff",
        "logo": "https://example.com/stale.png",
        "logos": [
            "https://a.espncdn.com/alabama.png",
            "https://a.espncdn.com/alabama-dark.png",
        ],
    }
    out = branding_fields(team)
    assert out["abbreviation"] == "ALA"
    assert out["color"] == "#9E1B32"
    assert out["alt_color"] == "#FFFFFF"
    assert out["logo"] == "https://a.espncdn.com/alabama.png"


def test_branding_fields_falls_back_to_logo_and_alt_color_keys():
    team = {
        "abbreviation": "UGA",
        "color": "#BA0C2F",
        "alt_color": "000000",
        "logo": "https://example.com/uga.png",
    }
    out = branding_fields(team)
    assert out["color"] == "#BA0C2F"
    assert out["alt_color"] == "#000000"
    assert out["logo"] == "https://example.com/uga.png"


def test_branding_fields_rejects_invalid_color():
    out = branding_fields({"color": "crimson", "logos": []})
    assert out["color"] is None
    assert out["logo"] is None


def test_fetch_fbs_teams_persists_branding(monkeypatch):
    monkeypatch.setattr(
        "src.collect.get_json",
        lambda *args, **kwargs: [
            {
                "id": 333,
                "school": "Alabama",
                "abbreviation": "ALA",
                "conference": "SEC",
                "classification": "fbs",
                "color": "9e1b32",
                "altColor": "ffffff",
                "logos": ["https://cdn.example/alabama.png"],
                "location": {"city": "Tuscaloosa", "state": "AL"},
            }
        ],
    )
    frame = fetch_fbs_teams(2026)
    row = frame.iloc[0]
    assert int(row["team_id"]) == 333
    assert row["abbreviation"] == "ALA"
    assert row["color"] == "#9E1B32"
    assert row["alt_color"] == "#FFFFFF"
    assert row["logo"] == "https://cdn.example/alabama.png"
