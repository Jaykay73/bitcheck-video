from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from app.services.audio_analyzer import analyze_audio_deepfake
from app.services.audio_feature_extractor import FEATURE_NAMES
from app.services.audio_model_loader import load_audio_model


class MockRandomForest:
    classes_ = np.array([0, 1])
    feature_importances_ = np.array([0.01] * 9 + [0.09] + [0.01] * 16)

    def predict(self, dataframe):
        return np.array([0])

    def predict_proba(self, dataframe):
        return np.array([[0.84, 0.16]])


class MockStringModel:
    classes_ = np.array(["REAL", "FAKE"])

    def predict(self, dataframe):
        return np.array(["FAKE"])

    def predict_proba(self, dataframe):
        return np.array([[0.2, 0.8]])


def feature_dataframe() -> pd.DataFrame:
    return pd.DataFrame([[0.0] * len(FEATURE_NAMES)], columns=FEATURE_NAMES)


def test_mock_rf_classes_zero_one_maps_zero_to_fake(tmp_path: Path) -> None:
    model_path = tmp_path / "BitcheckDeepfake.joblib"
    joblib.dump(MockRandomForest(), model_path)

    load_result = load_audio_model(model_path=model_path)
    result = analyze_audio_deepfake(feature_dataframe(), model_result=load_result)

    assert result.checked is True
    assert result.model_found is True
    assert result.class_mapping == {"0": "fake", "1": "real"}
    assert result.predicted_label == "fake"
    assert result.fake_probability == 0.84
    assert result.real_probability == 0.16
    assert result.risk_score == 0.84


def test_predict_proba_path_works_with_string_classes(tmp_path: Path) -> None:
    model_path = tmp_path / "BitcheckDeepfake.joblib"
    joblib.dump(MockStringModel(), model_path)

    load_result = load_audio_model(model_path=model_path)
    result = analyze_audio_deepfake(feature_dataframe(), model_result=load_result)

    assert result.checked is True
    assert result.class_mapping == {"REAL": "real", "FAKE": "fake"}
    assert result.predicted_label == "fake"
    assert result.fake_probability == 0.8
    assert result.real_probability == 0.2


def test_missing_model_returns_warning_not_crash(tmp_path: Path) -> None:
    load_result = load_audio_model(model_path=tmp_path / "missing.joblib")
    result = analyze_audio_deepfake(feature_dataframe(), model_result=load_result)

    assert result.checked is False
    assert result.model_found is False
    assert result.risk_score is None
    assert result.warnings


def test_feature_importance_extracted_if_available(tmp_path: Path) -> None:
    model_path = tmp_path / "BitcheckDeepfake.joblib"
    joblib.dump(MockRandomForest(), model_path)

    load_result = load_audio_model(model_path=model_path)
    result = analyze_audio_deepfake(feature_dataframe(), model_result=load_result)

    assert result.feature_importance_top
    assert result.feature_importance_top[0] == {"feature": "mfcc4", "importance": 0.09}
