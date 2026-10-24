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
            model_path = manifest_path.parent / model_path.parent.name / model_path.name
        predictions.append(entry["weight"] * joblib.load(model_path).predict(frame[manifest["features"]]))
    values = np.sum(predictions, axis=0)
    if manifest["kind"] == "classification":
        values = values[:, 1] if manifest["metric"] in PROBABILITY else np.asarray(manifest["classes"])[values.argmax(axis=1)]
    pd.DataFrame({"prediction": values}).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
