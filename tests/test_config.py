from cfb_model import config


def test_pro_tier_ingest_defaults():
    assert config.INGEST_FULL_DEFAULT is True
    assert config.INGEST_WEATHER_DEFAULT is True
