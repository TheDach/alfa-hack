"""Check organizer Qwen connectivity without printing credentials."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from dotenv import load_dotenv
    from research.planner import QwenPlanner
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", default=str(ROOT / ".env"))
    parser.add_argument("--report", default=str(ROOT / "reports/api_check.json"))
    args = parser.parse_args()
    load_dotenv(args.env_file, override=False)
    planner = QwenPlanner()
    allowed = [{"family": "catboost", "depth": 6, "iterations": 500,
                "learning_rate": .065, "l2": 5, "balanced": True,
                "features": "basic", "target_transform": "none"}]
    proposal = planner.propose({"kind": "classification", "metric": "macro_f1",
                               "train_rows": 1000, "numeric_columns": ["amount", "duration"],
                               "class_counts": {"normal": 900, "rare": 100}}, [], allowed, 90)
    result = {"enabled": planner.enabled, "model": planner._model,
              "accepted": proposal is not None, "events": planner.events,
              "usage": planner.usage, "disabled_reason": planner.disabled_reason}
    output = Path(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if proposal else 1


if __name__ == "__main__":
    raise SystemExit(main())
