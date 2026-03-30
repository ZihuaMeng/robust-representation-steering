"""Score steered completions with a local LM judge."""

import argparse
import json
import os
from pathlib import Path

import torch

import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from judges.local_lm_judge import JudgeConfig, LocalLMJudge


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"


METRICS = ["fluency", "helpfulness", "safety"]


def _load_candidates(path, max_examples=None):
    rows = []
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if max_examples and len(rows) >= max_examples:
                break
    return rows


def _score_records(records, judge):
    scored = []
    for idx, row in enumerate(records, 1):
        prompt = row.get("prompt", "")
        base_resp = row.get("baseline_completion", "")
        steered_resp = row.get("steered_completion", "")

        baseline_scores = judge.score(prompt, base_resp)
        steered_scores = judge.score(prompt, steered_resp)
        diff = {m: steered_scores.get(m, 0.0) - baseline_scores.get(m, 0.0) for m in METRICS}

        scored.append({
            **row,
            "baseline_scores": baseline_scores,
            "steered_scores": steered_scores,
            "score_diff": diff,
        })

        if idx % 5 == 0 or idx == len(records):
            print(f"Scored {idx}/{len(records)} examples")

    return scored


def _aggregate(scored):
    n = len(scored)
    if n == 0:
        return {}

    agg = {
        "n_examples": n,
        "baseline_means": {},
        "steered_means": {},
        "delta_means": {},
    }
    baseline_totals = {m: 0.0 for m in METRICS}
    steered_totals = {m: 0.0 for m in METRICS}
    delta_totals = {m: 0.0 for m in METRICS}
    wins = 0

    for row in scored:
        base_scores = row["baseline_scores"]
        steer_scores = row["steered_scores"]
        base_total = 0.0
        steer_total = 0.0
        for m in METRICS:
            base_val = float(base_scores.get(m, 0.0))
            steer_val = float(steer_scores.get(m, 0.0))
            baseline_totals[m] += base_val
            steered_totals[m] += steer_val
            delta = steer_val - base_val
            delta_totals[m] += delta
            base_total += base_val
            steer_total += steer_val
        if steer_total > base_total:
            wins += 1

    for m in METRICS:
        agg["baseline_means"][m] = baseline_totals[m] / n
        agg["steered_means"][m] = steered_totals[m] / n
        agg["delta_means"][m] = delta_totals[m] / n

    agg["win_rate"] = wins / n
    return agg


def run(args):
    candidates = _load_candidates(args.candidates, args.max_examples)
    if not candidates:
        raise RuntimeError("No steering candidates loaded")

    judge_cfg = JudgeConfig(
        model_name=args.judge_model,
        max_new_tokens=args.judge_max_new_tokens,
        temperature=args.judge_temperature,
        top_p=args.judge_top_p,
    )
    judge = LocalLMJudge(judge_cfg, device=args.judge_device, dtype=args.judge_dtype)

    scored = _score_records(candidates, judge)
    summary = _aggregate(scored)
    summary["method"] = args.method

    method_dir = OUTPUT_DIR / "evaluate" / args.method
    method_dir.mkdir(parents=True, exist_ok=True)
    data_path = method_dir / "steering_data.jsonl"
    summary_path = method_dir / "steering_summary.json"

    with open(data_path, "w") as f:
        for row in scored:
            f.write(json.dumps(row) + "\n")

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved per-example scores to {data_path}")
    print(f"Summary: {summary}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate steering completions with a local judge.")
    parser.add_argument("--candidates", required=True, help="Path to steering JSONL produced by run_steering_inference.py")
    parser.add_argument("--method", required=True, help="Method label (e.g., robust)")
    parser.add_argument("--max-examples", type=int, default=None, help="Optional cap on number of rows to score")

    parser.add_argument("--judge-model", required=True, help="HF model ID for the local judge")
    parser.add_argument("--judge-device", default=None, help="Optional device map override (e.g., cuda:0)")
    parser.add_argument("--judge-dtype", default="auto", help="Torch dtype string for the judge (default=auto)")
    parser.add_argument("--judge-max-new-tokens", type=int, default=64, help="Judge completion length")
    parser.add_argument("--judge-temperature", type=float, default=0.0, help="Judge sampling temperature")
    parser.add_argument("--judge-top-p", type=float, default=0.9, help="Judge top-p")
    return parser.parse_args()


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    args = parse_args()
    run(args)
