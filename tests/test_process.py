import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

import pandas as pd

from test_core import fixture_task
from test_core_review import runner_args
from research.engine import run


ROOT = Path(__file__).resolve().parents[1]


def test_worker_timeout_preserves_valid_submission(tmp_path, monkeypatch):
    task = fixture_task(tmp_path / "task", "regression")
    real_popen = subprocess.Popen
    def sleeping_worker(command, **kwargs):
        return real_popen([sys.executable, "-c", "import time; time.sleep(60)"], **kwargs)
    monkeypatch.setattr(subprocess, "Popen", sleeping_worker)
    report = run(runner_args(task.path, max_trials=1, trial_timeout=2.1), time.monotonic())
    frame = pd.read_csv(task.output)
    assert frame.id.tolist() == [70, 71]
    assert frame[task.target].notna().all()
    report = json.loads((Path(report["run_dir"]) / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "fallback"
    assert report["history"][0]["status"] == "timeout"


def test_archive_runs_from_another_working_directory(tmp_path):
    archive = ROOT / "dist" / "submission.zip"
    if not archive.exists():
        return
    unpacked = tmp_path / "unpacked"
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(unpacked)
    task = fixture_task(tmp_path / "input", "classification", True)
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    result = subprocess.run([sys.executable, str(unpacked / "solution.py"), str(task.path), "--offline",
                             "--time-budget", "30", "--max-trials", "0"],
                            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=40)
    assert result.returncode == 0, result.stderr
    result = pd.read_csv(task.output)
    assert result.id.tolist() == [70, 71]
    assert set(result[task.target]) <= {10, 20, 30}
