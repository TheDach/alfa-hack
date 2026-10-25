import io
import json
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from research.planner import QwenPlanner
from research.task import Task, profile_task, summarize_description


def make_task():
    count = 120
    train = pd.DataFrame({
        "surface_m2": np.tile(np.arange(10, dtype=float), 12),
        "construction_code": np.tile([1980., 1990., 2000.], 40),
        "region_code": np.tile(["north", "south"], 60),
        "kind_code": np.tile(["office", "home", "shop"], 40),
        "period_number": np.repeat(np.arange(1., 6.), 24),
        "outcome": np.arange(count, dtype=float),
    })
    test = train.drop(columns="outcome").iloc[:20].copy()
    test["period_number"] = 6.
    dictionary = pd.DataFrame([
        {"feature": "surface_m2", "dtype": "float", "role": "feature", "unit": "m2", "description_ru": "Площадь объекта"},
        {"feature": "construction_code", "dtype": "float", "role": "feature", "unit": "year", "description_ru": "Год постройки объекта"},
        {"feature": "region_code", "dtype": "category", "role": "feature", "description_ru": "Регион местоположения"},
        {"feature": "kind_code", "dtype": "category", "role": "feature", "description_ru": "Назначение объекта"},
        {"feature": "period_number", "dtype": "float", "role": "feature", "description_ru": "Код периода наблюдения"},
        {"feature": "outcome", "dtype": "float", "role": "target", "unit": "kg per year", "description_ru": "Измеренное годовое потребление"},
    ])
    description = ("# Forecast consumption\n\nPredict outcome from object measurements.\n\n" +
                   "Background with no modelling requirements. " * 400 +
                   "\n\nPrimary metric: RMSE on original target scale.\n\n" +
                   "Temporal validation must follow the observed period order.")
    return Task(Path("task.md"), description, train, test,
                pd.DataFrame({"outcome": np.zeros(len(test))}), dictionary,
                "outcome", [], "regression", "rmse", Path("submission.csv"),
                ["region_code", "kind_code"])


def test_profile_contains_units_semantics_and_late_requirements():
    profile = profile_task(make_task())
    assert "Primary metric: RMSE" in profile["description"]
    assert "Temporal validation" in profile["description"]
    assert len(profile["description"]) <= 3000
    by_name = {row["feature"]: row for row in profile["data_dictionary"]}
    assert profile["data_dictionary"][0]["feature"] == "outcome"
    assert by_name["outcome"]["unit"] == "kg per year"
    assert by_name["surface_m2"]["description"] == "Площадь объекта"
    assert "Background" not in summarize_description(make_task().description, 400)


def test_profile_detects_input_only_time_and_repeated_entity_structure():
    task = make_task()
    profile = profile_task(task)
    assert profile["temporal_shift"] == [{"column": "period_number", "train_min": 1.,
        "train_max": 5., "test_min": 6., "test_max": 6., "train_periods": 5}]
    assert 1 <= len(profile["entity_groups"]) <= 3
    for group in profile["entity_groups"]:
        assert {"surface_m2", "construction_code"} <= set(group["columns"])
        assert group["train_duplicate_fraction"] >= .2
        assert 0 <= group["test_seen_fraction"] <= 1
        assert "outcome" not in group["columns"]
    task.train["outcome"] = task.train["outcome"].iloc[::-1].to_numpy()
    changed = profile_task(task)
    assert changed["entity_groups"] == profile["entity_groups"]
    assert changed["temporal_shift"] == profile["temporal_shift"]


def test_bounded_semantic_profile_preserves_target_units_and_full_numeric_schema():
    profile = profile_task(make_task())
    profile["data_dictionary"] += [{"feature": f"other_{i}", "unit": "arbitrary units",
        "description": "Длинное описание измеряемого признака. " * 100} for i in range(100)]
    compact = QwenPlanner._bounded_profile(profile, 10_000)
    assert len(QwenPlanner._dump(compact).encode("utf-8")) <= 10_000
    assert compact["numeric_columns"] == profile["numeric_columns"]
    assert compact["temporal_shift"] == profile["temporal_shift"]
    assert compact["entity_groups"] == profile["entity_groups"]
    assert "Primary metric: RMSE" in compact["description"]
    assert compact["data_dictionary"][0]["unit"] == "kg per year"
    assert compact["dictionary_rows_omitted"] > 0


def test_new_model_configuration_reaches_planner_with_canonical_parameters():
    env = {"OPENAI_API_KEY": "test-only-key", "OPENAI_BASE_URL": "https://example.test/v1"}
    allowed = [{"family": "xgboost", "features": "interactions", "depth": 6,
        "iterations": 400, "learning_rate": .05, "l2": 4,
        "min_child_samples": 15, "num_leaves": 63, "max_bin": 255,
        "subsample": .9, "colsample_bytree": .8, "balance_power": .5,
        "group_features": [["surface_m2", "construction_code"]]}]
    body = {"choices": [{"message": {"content": json.dumps({"config_id": 0,
        "hypothesis": "Compare grouped object history under the period shift."})}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 30, "total_tokens": 130}}
    with patch.dict(os.environ, env, clear=True), patch("urllib.request.urlopen",
            return_value=io.BytesIO(json.dumps(body).encode())) as call:
        selected = QwenPlanner().propose(profile_task(make_task()), [], allowed, 60)
    assert selected["family"] == "xgboost"
    assert selected["group_features"] == allowed[0]["group_features"]
    payload = json.loads(json.loads(call.call_args.args[0].data)["messages"][1]["content"])
    assert payload["allowed_configurations"][0]["colsample_bytree"] == .8
    invalid = {**allowed[0], "group_features": [["surface_m2", "outcome"]]}
    with patch.dict(os.environ, env, clear=True), patch("urllib.request.urlopen") as call:
        assert QwenPlanner().propose(profile_task(make_task()), [], [invalid], 60) is None
        call.assert_not_called()

def test_actual_public_regression_profile_preserves_time_shift_and_entity_groups():
    from research.task import load_task
    import pytest
    path = Path("C:/Users/vlad/Desktop/hackaton/data/regression/task.md")
    if not path.is_file():
        pytest.skip("Public development data unavailable")
    profile = profile_task(load_task(path))
    periods = {row["column"]: row for row in profile["temporal_shift"]}
    assert periods["Year_Factor"]["train_max"] == 5
    assert periods["Year_Factor"]["test_min"] == 6
    assert any(set(row["columns"]) == {"floor_area", "year_built", "facility_type", "State_Factor"}
               for row in profile["entity_groups"])
    bounded = QwenPlanner._bounded_profile(profile, 11_160)
    assert bounded["temporal_shift"] == profile["temporal_shift"]
    assert bounded["entity_groups"] == profile["entity_groups"]
    assert bounded["numeric_columns"] == profile["numeric_columns"]
    by_name = {row["feature"]: row for row in bounded["data_dictionary"]}
    assert "site_eui" in by_name
    assert "floor_area" in by_name and by_name["floor_area"]["unit"]
    assert "year_built" in by_name and "description" in by_name["year_built"]
    assert len(QwenPlanner._dump(bounded).encode("utf-8")) <= 11_160


def test_history_residual_requires_valid_regression_group_configuration():
    config = {"family": "lightgbm", "history_residual": True,
              "group_features": [["surface_m2", "construction_code"]]}
    assert QwenPlanner._configuration(config) == config
    assert QwenPlanner._configuration({"family": "lightgbm", "history_residual": True}) is None
    assert QwenPlanner._configuration({**config, "history_residual": 1}) is None
    assert QwenPlanner._valid_groups(config, profile_task(make_task()))
    assert not QwenPlanner._valid_groups(config, {"kind": "classification",
        "numeric_columns": ["surface_m2", "construction_code"]})
