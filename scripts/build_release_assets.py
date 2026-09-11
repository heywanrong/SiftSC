"""Build bundled gate profiles and README assets from the paper's real records.

This release script intentionally consumes the original paper repository through
an explicit argument. The product repository stores only small trained profiles,
aggregate tables, plots, and provenance checksums—not model weights or raw traces.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from siftsc.calibration import fit_logistic
from siftsc.features import FEATURE_NAMES, prompt_features

PROFILE_SPECS = {
    "qwen05b-fp16-logistic": ("qwen05b_fp16", "logistic", 0.625),
    "qwen05b-q4-confidence": ("qwen05b_q4", "confidence", 0.915),
    "gemma3-1b-fp16-logistic": ("gemma3_1b_fp16", "logistic", 0.460),
    "gemma3-1b-q4-logistic": ("gemma3_1b_q4", "logistic", 0.515),
}

DISPLAY_NAMES = {
    "qwen05b_fp16": "Qwen-0.5B FP16",
    "qwen05b_q4": "Qwen-0.5B Q4",
    "gemma3_1b_fp16": "Gemma-1B FP16",
    "gemma3_1b_q4": "Gemma-1B Q4",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_prompts(root: Path) -> dict[str, str]:
    prompts: dict[str, str] = {}
    for relative in ("data/gsm8k_eval.jsonl", "data/gsm8k_calib.jsonl", "data/math500.jsonl"):
        for item in read_jsonl(root / relative):
            prompts[item["id"]] = item.get("question") or item.get("problem") or ""
    return prompts


def record_vector(record: dict[str, Any], prompt: str) -> np.ndarray:
    values = {
        "greedy_conf": float(record.get("greedy_conf") or 0.0),
        "greedy_top5_ent": float(record.get("greedy_top5_ent") or 0.0),
        "greedy_mean_ent": float(record.get("greedy_mean_ent") or 0.0),
        "greedy_n_tokens": float(record.get("greedy_n_tokens") or 0.0),
        **prompt_features(prompt),
    }
    return np.asarray([values[name] for name in FEATURE_NAMES], dtype=np.float64)


def write_profiles(root: Path, output: Path, prompts: dict[str, str]) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    output.mkdir(parents=True, exist_ok=True)
    for profile_name, (cell, gate_type, threshold) in PROFILE_SPECS.items():
        source = root / "runs/block_a_combined" / cell / "combined/seed_1/records.jsonl"
        records = read_jsonl(source)
        features = np.stack(
            [record_vector(record, prompts.get(record["id"], "")) for record in records]
        )
        labels = np.asarray(
            [int(record["sc5_correct"] and not record["greedy_correct"]) for record in records],
            dtype=np.int64,
        )
        provenance = {
            "paper": "When Self-Consistency Hurts (ICONIP 2026, forthcoming)",
            "cell": cell,
            "dataset": "GSM8K-200 + MATH-500-200",
            "n": len(records),
            "positive_rate": float(labels.mean()),
            "source_sha256": sha256(source),
            "threshold_selection": (
                "5-fold out-of-fold, max skip rate subject to >=98% empirical SC accuracy retention"
            ),
            "note": (
                "Threshold was selected on out-of-fold scores; the serialized model "
                "was refit on all 400 prompts."
            ),
        }
        if gate_type == "confidence":
            confidence = features[:, FEATURE_NAMES.index("greedy_conf")]
            payload = {
                "schema_version": 1,
                "name": profile_name,
                "gate_type": "confidence",
                "threshold": threshold,
                "confidence_min": float(confidence.min()),
                "confidence_max": float(confidence.max()),
                "provenance": provenance,
            }
        else:
            payload = fit_logistic(features, labels).as_json(
                name=profile_name,
                threshold=threshold,
                provenance=provenance,
            )
        destination = output / f"{profile_name}.json"
        destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        manifest.append(
            {"profile": profile_name, **provenance, "artifact_sha256": sha256(destination)}
        )
    return manifest


def aggregate_task(root: Path, cell: str, dataset: str) -> dict[str, Any]:
    source_root = "runs/block_a_b_v1" if dataset == "gsm8k" else "runs/block_a_math"
    path = root / source_root / cell / dataset / "seed_1/records.jsonl"
    records = read_jsonl(path)
    greedy = np.asarray([record["greedy_correct"] for record in records], dtype=float)
    sc = np.asarray([record["sc5_correct"] for record in records], dtype=float)
    help_rate = np.mean((sc == 1) & (greedy == 0))
    harm_rate = np.mean((sc == 0) & (greedy == 1))
    return {
        "cell": DISPLAY_NAMES[cell],
        "dataset": "GSM8K" if dataset == "gsm8k" else "MATH-500",
        "n": len(records),
        "greedy_accuracy": float(greedy.mean()),
        "sc5_accuracy": float(sc.mean()),
        "help_rate": float(help_rate),
        "harm_rate": float(harm_rate),
        "sc_net_gain_points": float(100 * (sc.mean() - greedy.mean())),
        "source_sha256": sha256(path),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_benchmarks(root: Path, output: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    task_rows = [
        aggregate_task(root, cell, dataset)
        for dataset in ("gsm8k", "math500")
        for cell in DISPLAY_NAMES
    ]
    write_csv(output / "per_task.csv", task_rows)

    summary_path = root / "runs/ttc_gate_combined_consolidated/summary.json"
    summaries = json.loads(summary_path.read_text(encoding="utf-8"))
    pooled_rows: list[dict[str, Any]] = []
    for item in summaries:
        profile_name = next(name for name, spec in PROFILE_SPECS.items() if spec[0] == item["cell"])
        method = PROFILE_SPECS[profile_name][1]
        boot = item["boot"]["conf_only" if method == "confidence" else "logistic"]
        pooled_rows.append(
            {
                "cell": DISPLAY_NAMES[item["cell"]],
                "gate": method,
                "n": item["n"],
                "skip_rate": boot["skip_rate"]["mean"],
                "skip_ci_low": boot["skip_rate"]["ci_lo"],
                "skip_ci_high": boot["skip_rate"]["ci_hi"],
                "cost_ratio": boot["cost_ratio"]["mean"],
                "accuracy_retention": boot["acc_retention"]["mean"],
                "retention_ci_low": boot["acc_retention"]["ci_lo"],
                "retention_ci_high": boot["acc_retention"]["ci_hi"],
                "source_sha256": sha256(summary_path),
            }
        )
    write_csv(output / "pooled_gate.csv", pooled_rows)
    return task_rows, pooled_rows


def plot_help_harm(rows: list[dict[str, Any]], destination: Path) -> None:
    labels = [f"{row['cell']}\n{row['dataset']}" for row in rows]
    help_values = [100 * row["help_rate"] for row in rows]
    harm_values = [-100 * row["harm_rate"] for row in rows]
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.barh(y, help_values, color="#2563eb", label="SC helps (wrong → right)")
    ax.barh(y, harm_values, color="#f97316", label="SC hurts (right → wrong)")
    ax.axvline(0, color="#0f172a", linewidth=0.8)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Share of prompts (%)")
    ax.set_title("Self-consistency helps—and hurts—small models", loc="left", weight="bold")
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.10))
    ax.grid(axis="x", alpha=0.18)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(destination, dpi=180, facecolor="white")
    plt.close(fig)


def plot_pooled(rows: list[dict[str, Any]], destination: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    for index, row in enumerate(rows):
        x = row["cost_ratio"]
        y = 100 * row["accuracy_retention"]
        # The bounds reverse under C=1+4*(1-skip); reorder explicitly.
        cost_low = 1 + 4 * (1 - row["skip_ci_high"])
        cost_high = 1 + 4 * (1 - row["skip_ci_low"])
        ax.errorbar(
            x,
            y,
            xerr=[[x - cost_low], [cost_high - x]],
            fmt="o",
            markersize=9,
            capsize=4,
            color=("#2563eb", "#7c3aed", "#059669", "#ea580c")[index],
            label=row["cell"],
        )
    ax.scatter([1], [100], marker="D", color="#64748b", label="greedy cost reference")
    ax.scatter([5], [100], marker="s", color="#111827", label="always-SC cost reference")
    ax.axhline(100, color="#64748b", linestyle="--", linewidth=1)
    ax.set_xlim(0.8, 5.2)
    ax.set_xlabel("Generation-pass cost (\u00d7 greedy)")
    ax.set_ylabel("Accuracy retention vs always-SC (%)")
    ax.set_title(
        "Measured pooled operating points (n=400 per model/precision)",
        loc="left",
        weight="bold",
    )
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(destination, dpi=180, facecolor="white")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    prompts = load_prompts(args.paper_root)
    profiles = write_profiles(args.paper_root, args.repo_root / "src/siftsc/profiles", prompts)
    task_rows, pooled_rows = write_benchmarks(args.paper_root, args.repo_root / "docs/benchmarks")
    assets = args.repo_root / "docs/assets"
    assets.mkdir(parents=True, exist_ok=True)
    plot_help_harm(task_rows, assets / "help-harm.png")
    plot_pooled(pooled_rows, assets / "pooled-operating-points.png")
    (args.repo_root / "docs/benchmarks/profile_manifest.json").write_text(
        json.dumps(profiles, indent=2) + "\n", encoding="utf-8"
    )
    print(f"built {len(profiles)} profiles, {len(task_rows)} task rows, and 2 figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
