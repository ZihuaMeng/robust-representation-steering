"""Aggregate steering evaluation tables."""

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


METRICS = ["fluency", "helpfulness", "safety"]


def _read_jsonl(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _flatten_row(row):
    flat = {k: v for k, v in row.items() if k not in {"baseline_scores", "steered_scores", "score_diff"}}
    base_scores = row.get("baseline_scores", {})
    steer_scores = row.get("steered_scores", {})
    for m in METRICS:
        flat[f"baseline_{m}"] = float(base_scores.get(m, 0.0))
        flat[f"steered_{m}"] = float(steer_scores.get(m, 0.0))
        flat[f"delta_{m}"] = flat[f"steered_{m}"] - flat[f"baseline_{m}"]
    flat["steered_total"] = sum(flat[f"steered_{m}"] for m in METRICS)
    return flat


def _load_dataframe(evaluate_dir: Path) -> pd.DataFrame:
    frames = []
    for method_dir in evaluate_dir.iterdir():
        data_path = method_dir / "steering_data.jsonl"
        if not data_path.exists():
            continue
        rows = _read_jsonl(data_path)
        if not rows:
            continue
        flat_rows = [_flatten_row(r) for r in rows]
        df = pd.DataFrame(flat_rows)
        df["method"] = method_dir.name
        frames.append(df)

    if not frames:
        raise RuntimeError(f"No steering_data.jsonl files found under {evaluate_dir}")

    return pd.concat(frames, ignore_index=True)


def _write_by_method(df: pd.DataFrame, out_path: Path):
    cols = [f"steered_{m}" for m in METRICS] + [f"delta_{m}"]
    summary = df.groupby("method")[[c for c in cols]].mean().reset_index()
    summary.to_csv(out_path, index=False)


def _write_by_factor(df: pd.DataFrame, out_path: Path):
    if "factor" not in df.columns:
        return
    cols = ["method", "factor"] + [f"steered_{m}" for m in METRICS]
    df.groupby(["method", "factor"])[[f"steered_{m}" for m in METRICS]].mean().reset_index().to_csv(out_path, index=False)


def _write_best_method(df: pd.DataFrame, out_path: Path):
    if "example_id" not in df.columns:
        return
    idx = df.sort_values("steered_total", ascending=False).groupby("example_id").head(1)
    idx[["example_id", "method", "steered_total"]].to_csv(out_path, index=False)


def main(args):
    evaluate_dir = Path(args.evaluate_dir)
    df = _load_dataframe(evaluate_dir)

    _write_by_method(df, evaluate_dir / "steering_metrics_by_method.csv")
    _write_by_factor(df, evaluate_dir / "steering_metrics_by_factor.csv")
    _write_best_method(df, evaluate_dir / "steering_best_method_per_example.csv")
    print(f"Aggregates written under {evaluate_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate steering evaluation tables")
    parser.add_argument("--evaluate-dir", default=str(ROOT / "outputs" / "evaluate"),
                        help="Directory containing per-method evaluation subfolders")
    args = parser.parse_args()
    main(args)
