import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd

from research.metrics import PROBABILITY


def main():
    parser = argparse.ArgumentParser(description="Predict with saved final models")
    parser.add_argument("manifest")
    parser.add_argument("input")
    parser.add_argument("output")
    args = parser.parse_args()
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frame = pd.read_csv(args.input)
    predictions = []
    for entry in manifest["models"]:
        model_path = Path(entry["path"])
        if not model_path.exists():
            # Preserve nested fold paths when a run directory was moved.
            candidates = [manifest_path.parent / Path(*model_path.parts[i:]) for i in range(len(model_path.parts))]
            matches = [candidate for candidate in candidates if candidate.is_file()]
            if len(matches) != 1:
                raise FileNotFoundError(f"Cannot resolve saved model: {entry['path']}")
            model_path = matches[0]
        predictions.append(entry["weight"] * joblib.load(model_path).predict(frame[manifest["features"]]))
    values = np.sum(predictions, axis=0)
    if manifest["kind"] == "classification":
        values = values[:, 1] if manifest["metric"] in PROBABILITY else np.asarray(manifest["classes"])[values.argmax(axis=1)]
    if manifest["kind"] == "regression" and manifest["metric"] == "rmsle":
        values = np.maximum(values, 0)
    pd.DataFrame({"prediction": values}).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
