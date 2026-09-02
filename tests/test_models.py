from sklearn.metrics import mean_absolute_error

from cfb_model.evaluate import walk_forward
from cfb_model.features import TARGET_COL, build_features
from cfb_model.models import completed_mask, fit_models, predict_frame, save_models, load_models
from cfb_model.store import connect


def test_models_fit_and_beat_mean_baseline(db_path):
    conn = connect(db_path)
    try:
        frame = build_features(conn)
    finally:
        conn.close()
    train = frame.loc[completed_mask(frame)]
    models = fit_models(train)
    pred = predict_frame(models, train, apply_learning=False)
    model_mae = mean_absolute_error(train[TARGET_COL], pred["pred_margin"])
    baseline = mean_absolute_error(train[TARGET_COL], [train[TARGET_COL].mean()] * len(train))
    assert model_mae < baseline
    assert pred["pred_home_wp"].between(0, 1).all()


def test_walk_forward_and_persist(db_path, tmp_path, monkeypatch):
    monkeypatch.setattr("cfb_model.config.MODELS_DIR", tmp_path / "models")
    conn = connect(db_path)
    try:
        frame = build_features(conn)
    finally:
        conn.close()
    scored, summary, models = walk_forward(frame, min_test_year=2023, min_train=5)
    assert not scored.empty
    assert summary["overall"]["model_mae"] > 0
    save_models(models, tmp_path / "models")
    loaded = load_models(tmp_path / "models")
    assert loaded["columns"] == models["columns"]
    again = predict_frame(loaded, scored.head(5), apply_learning=False)
    assert "pred_margin" in again
