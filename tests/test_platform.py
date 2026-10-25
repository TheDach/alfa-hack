import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
import pandas as pd
import pytest

from research.features import CategoricalEncoder
from research.planner import QwenPlanner
from tools.build_submission import build

ROOT = Path(__file__).resolve().parents[1]


def test_native_category_vocabulary_is_fit_only_and_names_are_safe():
    fit = pd.DataFrame({"[nominal]": ["red", "blue", "red"], "value": [1., np.nan, 3.]})
    encoder = CategoricalEncoder().fit(fit)
    test = encoder.transform(pd.DataFrame({"[nominal]": ["new", "blue"], "value": [np.nan, 8.]}))
    assert test.columns.tolist() == ["f0", "f1"]
    assert test.f0.cat.categories.tolist() == ["red", "blue"]
    assert pd.isna(test.f0.iloc[0])
    assert test.f0.iloc[1] == "blue"
    assert pd.isna(test.f1.iloc[0])


def test_platform_does_not_need_model_environment_variable(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "unit-test-placeholder")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test/v1")
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    planner = QwenPlanner()
    assert planner.enabled
    assert planner._model == "qwen/qwen3.8-27b"


@pytest.mark.parametrize("kind", ["regression", "classification"])
def test_flat_archive_copier_runs_real_worker_and_final_fit(tmp_path, kind):
    archive = build()
    task = tmp_path / "task"
    task.mkdir()
    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
        assert all("/" not in name and chr(92) not in name for name in names)
        assert bundle.read("requirements.txt") == (ROOT / "reference/requirements.txt").read_bytes()
        assert all(name.endswith(".py") or name == "requirements.txt" for name in names)
        # Replicate the platform: copy only eligible root files to the task dir.
        for name in names:
            if name.endswith(".py") or name == "requirements.txt":
                (task / name).write_bytes(bundle.read(name))
    rng = np.random.default_rng(42)
    x = rng.normal(size=240)
    y = 4 * x + rng.normal(size=240) * .05
    if kind == "classification":
        y = np.where(x < -.5, "low", np.where(x > .5, "high", "middle"))
    pd.DataFrame({"id": np.arange(240), "x": x, "segment": np.tile(["a", "b"], 120),
                  "outcome": y}).to_csv(task / "train.csv", index=False)
    pd.DataFrame({"id": [900, 901], "x": [-1., 1.], "segment": ["a", "unseen"]}).to_csv(task / "test.csv", index=False)
    pd.DataFrame({"id": [900, 901], "outcome": [0., 0.]}).to_csv(task / "sample_submission.csv", index=False)
    pd.DataFrame({"feature": ["id", "x", "segment", "outcome"],
                  "role": ["id", "feature", "feature", "target"],
                  "dtype": ["integer", "float", "category", "float"]}).to_csv(task / "data_dictionary.csv", index=False)
    description = "Regression. Primary metric: RMSE." if kind == "regression" else "Classification. Primary metric: Macro F1."
    (task / "task.md").write_text(description, encoding="utf-8")
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    result = subprocess.run([sys.executable, str(task / "solution.py"), "task.md", "--offline",
                             "--time-budget", "70", "--max-trials", "1", "--threads", "1",
                             "--trial-timeout", "25", "--memory-gb", "2"],
                            cwd=task, env=env, capture_output=True, text=True, timeout=85)
    assert result.returncode == 0, result.stdout + result.stderr
    latest = json.loads((task / "research_runs/latest.json").read_text())
    report = json.loads(Path(latest["report"]).read_text(encoding="utf-8"))
    assert report["status"] == "complete", report
    assert report["history"][0]["status"] == "ok"
    prediction = pd.read_csv(task / "submission.csv")
    assert prediction.id.tolist() == [900, 901]
    if kind == "regression":
        assert prediction.outcome.iloc[0] < prediction.outcome.iloc[1]
    else:
        assert prediction.outcome.tolist() == ["low", "high"]
    if kind == "regression":
        assert np.isfinite(prediction.outcome).all()
    else:
        assert set(prediction.outcome) <= {"low", "middle", "high"}
