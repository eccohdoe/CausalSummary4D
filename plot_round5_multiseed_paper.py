from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, pstdev

import matplotlib.pyplot as plt
import numpy as np


METHODS = [
    ("A", "Old target"),
    ("C", "Task-aligned target"),
]

COLORS = {
    "A": "#8A9199",
    "C": "#7F8F84",
    "pooled": "#B7A99A",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate round-5 matched A/C multi-seed results.")
    parser.add_argument("--a-runs", nargs="+", type=Path, required=True)
    parser.add_argument("--c-runs", nargs="+", type=Path, required=True)
    parser.add_argument("--a-probes", nargs="+", type=Path, required=True)
    parser.add_argument("--c-probes", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def seed_from_name(path: Path) -> int:
    name = path.name
    if "seed1" in name:
        return 1
    if "seed2" in name:
        return 2
    return 0


def best_row(run_dir: Path) -> dict:
    summary = load_json(run_dir / "summary.json")
    history = load_json(run_dir / "history.json")
    return history[summary["best_epoch"]], summary


def pair_runs_and_probes(run_dirs: list[Path], probe_paths: list[Path], method_key: str) -> list[dict]:
    probe_by_seed = {seed_from_name(path.parent): path for path in probe_paths}
    entries = []
    for run_dir in sorted(run_dirs, key=seed_from_name):
        seed = seed_from_name(run_dir)
        best_hist, summary = best_row(run_dir)
        probe = load_json(probe_by_seed[seed])
        entry = {
            "seed": seed,
            "method": method_key,
            "run_dir": str(run_dir),
            "probe_path": str(probe_by_seed[seed]),
            "best_val_video_acc1": summary["best_val_video_acc1"],
            "final_val_video_acc1": summary["final_val_video_acc1"],
            "best_epoch": summary["best_epoch"],
            "causality_passed": summary["causality"]["passed"],
            "causality_max_abs_logit_diff": summary["causality"]["max_abs_logit_diff"],
            "summary_probe_acc": probe["classification"]["summary_final"]["mlp"]["best_val_video_acc"],
            "pooled_probe_acc": probe["classification"]["pooled_final"]["mlp"]["best_val_video_acc"],
            "summary_delta_mse": probe["delta_regression"]["summary"]["mlp"]["best_val_mse"],
            "pooled_delta_mse": probe["delta_regression"]["pooled_state"]["mlp"]["best_val_mse"],
            "projected_pooled_target_delta_mse": probe["delta_regression"].get("projected_pooled_target", {}).get("mlp", {}).get("best_val_mse"),
            "pred_target_mse_val": probe["predictability"]["pred_target_mse_h1_checkpoint_val"],
            "summary_std": best_hist.get("summary_std"),
            "summary_std_min": best_hist.get("summary_std_min"),
            "slot_corr": best_hist.get("slot_corr"),
            "slot_max_corr": best_hist.get("slot_max_corr"),
        }
        entries.append(entry)
    return entries


def format_pct(value: float) -> str:
    return f"{100.0 * value:.2f}"


def format_mean_std(values: list[float], scale: float = 100.0, digits: int = 2) -> str:
    mu = mean(values) * scale
    sigma = pstdev(values) * scale
    return f"{mu:.{digits}f} ± {sigma:.{digits}f}"


def write_csv(path: Path, headers: list[str], rows: list[list[object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def write_md(path: Path, headers: list[str], rows: list[list[object]]) -> None:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def aggregate(entries: list[dict]) -> dict:
    return {
        "best_val_video_acc1_mean": mean(x["best_val_video_acc1"] for x in entries),
        "best_val_video_acc1_std": pstdev(x["best_val_video_acc1"] for x in entries),
        "final_val_video_acc1_mean": mean(x["final_val_video_acc1"] for x in entries),
        "final_val_video_acc1_std": pstdev(x["final_val_video_acc1"] for x in entries),
        "summary_probe_acc_mean": mean(x["summary_probe_acc"] for x in entries),
        "summary_probe_acc_std": pstdev(x["summary_probe_acc"] for x in entries),
        "pooled_probe_acc_mean": mean(x["pooled_probe_acc"] for x in entries),
        "pooled_probe_acc_std": pstdev(x["pooled_probe_acc"] for x in entries),
        "summary_delta_mse_mean": mean(x["summary_delta_mse"] for x in entries),
        "pooled_delta_mse_mean": mean(x["pooled_delta_mse"] for x in entries),
        "projected_pooled_target_delta_mse_mean": mean(
            x["projected_pooled_target_delta_mse"] for x in entries if x["projected_pooled_target_delta_mse"] is not None
        ) if any(x["projected_pooled_target_delta_mse"] is not None for x in entries) else None,
        "pred_target_mse_val_mean": mean(x["pred_target_mse_val"] for x in entries),
        "summary_std_mean": mean(x["summary_std"] for x in entries),
        "summary_std_min_mean": mean(x["summary_std_min"] for x in entries),
        "slot_corr_mean": mean(x["slot_corr"] for x in entries),
        "slot_max_corr_mean": mean(x["slot_max_corr"] for x in entries),
    }


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.color": "#D7D3CF",
        "font.size": 10,
        "axes.titlesize": 14,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })


def plot_main(entries_by_method: dict[str, list[dict]], output_dir: Path) -> list[str]:
    apply_style()
    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    x_positions = {"A": 0, "C": 1}
    for seed in sorted({e["seed"] for entries in entries_by_method.values() for e in entries}):
        y_a = 100.0 * next(e["best_val_video_acc1"] for e in entries_by_method["A"] if e["seed"] == seed)
        y_c = 100.0 * next(e["best_val_video_acc1"] for e in entries_by_method["C"] if e["seed"] == seed)
        ax.plot([x_positions["A"], x_positions["C"]], [y_a, y_c], color="#C8C3BC", linewidth=1.2, alpha=0.9, zorder=1)
        ax.scatter([x_positions["A"], x_positions["C"]], [y_a, y_c], s=28, color=["#8A9199", "#7F8F84"], zorder=2)

    for method_key, _label in METHODS:
        vals = [100.0 * e["best_val_video_acc1"] for e in entries_by_method[method_key]]
        mu = mean(vals)
        sigma = pstdev(vals)
        ax.errorbar(
            [x_positions[method_key]],
            [mu],
            yerr=[sigma],
            fmt="o",
            markersize=8,
            color=COLORS[method_key],
            ecolor=COLORS[method_key],
            elinewidth=2,
            capsize=4,
            zorder=3,
        )
        ax.text(x_positions[method_key], mu + sigma + 0.25, f"{mu:.2f}", ha="center", va="bottom", fontsize=10)

    ax.set_xticks([0, 1], ["A", "C"])
    ax.set_ylabel("Best val video acc (%)")
    ax.set_title("Matched 3-seed MSR comparison")
    all_vals = [100.0 * e["best_val_video_acc1"] for entries in entries_by_method.values() for e in entries]
    ax.set_ylim(min(all_vals) - 1.5, max(all_vals) + 1.8)
    fig.tight_layout()
    png = output_dir / "round5_main_multiseed.png"
    pdf = output_dir / "round5_main_multiseed.pdf"
    fig.savefig(png, dpi=260, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [str(png), str(pdf)]


def plot_probe(entries_by_method: dict[str, list[dict]], output_dir: Path) -> list[str]:
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
    for ax, metric_key, title in [
        (axes[0], "summary_probe_acc", "Summary probe"),
        (axes[1], "pooled_probe_acc", "Pooled probe"),
    ]:
        for seed in sorted({e["seed"] for entries in entries_by_method.values() for e in entries}):
            y_a = 100.0 * next(e[metric_key] for e in entries_by_method["A"] if e["seed"] == seed)
            y_c = 100.0 * next(e[metric_key] for e in entries_by_method["C"] if e["seed"] == seed)
            ax.plot([0, 1], [y_a, y_c], color="#C8C3BC", linewidth=1.2, alpha=0.9, zorder=1)
            ax.scatter([0, 1], [y_a, y_c], s=26, color=["#8A9199", "#7F8F84"], zorder=2)

        for idx, (method_key, _label) in enumerate(METHODS):
            vals = [100.0 * e[metric_key] for e in entries_by_method[method_key]]
            mu = mean(vals)
            sigma = pstdev(vals)
            ax.errorbar([idx], [mu], yerr=[sigma], fmt="o", markersize=8, color=COLORS[method_key], ecolor=COLORS[method_key], elinewidth=2, capsize=4, zorder=3)
            ax.text(idx, mu + sigma + 0.2, f"{mu:.2f}", ha="center", va="bottom", fontsize=9)

        ax.set_xticks([0, 1], ["A", "C"])
        ax.set_title(title)
        ax.grid(axis="y")

    axes[0].set_ylabel("Video probe acc (%)")
    ymin = min(
        100.0 * e[key]
        for entries in entries_by_method.values()
        for e in entries
        for key in ["summary_probe_acc", "pooled_probe_acc"]
    )
    ymax = max(
        100.0 * e[key]
        for entries in entries_by_method.values()
        for e in entries
        for key in ["summary_probe_acc", "pooled_probe_acc"]
    )
    axes[0].set_ylim(ymin - 2.0, ymax + 2.0)
    fig.suptitle("Representation discriminativeness across seeds", y=1.02)
    fig.tight_layout()
    png = output_dir / "round5_probe_multiseed.png"
    pdf = output_dir / "round5_probe_multiseed.pdf"
    fig.savefig(png, dpi=260, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [str(png), str(pdf)]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    entries_by_method = {
        "A": pair_runs_and_probes(args.a_runs, args.a_probes, "A"),
        "C": pair_runs_and_probes(args.c_runs, args.c_probes, "C"),
    }
    aggregate_by_method = {method: aggregate(entries) for method, entries in entries_by_method.items()}

    seed_rows = []
    for method_key, label in METHODS:
        for entry in entries_by_method[method_key]:
            seed_rows.append([
                label,
                entry["seed"],
                format_pct(entry["best_val_video_acc1"]),
                format_pct(entry["final_val_video_acc1"]),
                entry["best_epoch"],
                "pass" if entry["causality_passed"] else "fail",
                format_pct(entry["summary_probe_acc"]),
                format_pct(entry["pooled_probe_acc"]),
                round(entry["summary_delta_mse"], 4),
                round(entry["pooled_delta_mse"], 4),
                "-" if entry["projected_pooled_target_delta_mse"] is None else round(entry["projected_pooled_target_delta_mse"], 4),
                round(entry["summary_std"], 4),
                round(entry["summary_std_min"], 4),
                round(entry["slot_corr"], 4),
                round(entry["slot_max_corr"], 4),
            ])

    aggregate_rows = []
    for method_key, label in METHODS:
        agg = aggregate_by_method[method_key]
        aggregate_rows.append([
            label,
            format_mean_std([e["best_val_video_acc1"] for e in entries_by_method[method_key]]),
            format_mean_std([e["final_val_video_acc1"] for e in entries_by_method[method_key]]),
            format_mean_std([e["summary_probe_acc"] for e in entries_by_method[method_key]]),
            format_mean_std([e["pooled_probe_acc"] for e in entries_by_method[method_key]]),
            f"{agg['summary_delta_mse_mean']:.4f}",
            f"{agg['pooled_delta_mse_mean']:.4f}",
            "-" if agg["projected_pooled_target_delta_mse_mean"] is None else f"{agg['projected_pooled_target_delta_mse_mean']:.4f}",
            f"{agg['summary_std_mean']:.4f}",
            f"{agg['summary_std_min_mean']:.4f}",
            f"{agg['slot_corr_mean']:.4f}",
            f"{agg['slot_max_corr_mean']:.4f}",
        ])

    paired_deltas = []
    for seed in sorted(e["seed"] for e in entries_by_method["A"]):
        a_entry = next(e for e in entries_by_method["A"] if e["seed"] == seed)
        c_entry = next(e for e in entries_by_method["C"] if e["seed"] == seed)
        paired_deltas.append({
            "seed": seed,
            "best_val_video_acc1_delta": c_entry["best_val_video_acc1"] - a_entry["best_val_video_acc1"],
            "summary_probe_delta": c_entry["summary_probe_acc"] - a_entry["summary_probe_acc"],
            "pooled_probe_delta": c_entry["pooled_probe_acc"] - a_entry["pooled_probe_acc"],
        })

    delta_rows = [
        [
            d["seed"],
            f"{100.0 * d['best_val_video_acc1_delta']:.2f}",
            f"{100.0 * d['summary_probe_delta']:.2f}",
            f"{100.0 * d['pooled_probe_delta']:.2f}",
        ]
        for d in paired_deltas
    ]
    delta_rows.append([
        "mean",
        f"{100.0 * mean(d['best_val_video_acc1_delta'] for d in paired_deltas):.2f}",
        f"{100.0 * mean(d['summary_probe_delta'] for d in paired_deltas):.2f}",
        f"{100.0 * mean(d['pooled_probe_delta'] for d in paired_deltas):.2f}",
    ])

    seed_headers = [
        "Method",
        "Seed",
        "Best val video acc",
        "Final val video acc",
        "Best epoch",
        "Causality",
        "Summary probe acc",
        "Pooled probe acc",
        "Summary delta MSE",
        "Pooled delta MSE",
        "Projected pooled-target delta MSE",
        "summary_std",
        "summary_std_min",
        "slot_corr",
        "slot_max_corr",
    ]
    agg_headers = [
        "Method",
        "Best val video acc mean±std",
        "Final val video acc mean±std",
        "Summary probe mean±std",
        "Pooled probe mean±std",
        "Summary delta MSE mean",
        "Pooled delta MSE mean",
        "Projected pooled-target delta MSE mean",
        "summary_std mean",
        "summary_std_min mean",
        "slot_corr mean",
        "slot_max_corr mean",
    ]
    delta_headers = ["Seed", "C-A best val delta (pp)", "C-A summary probe delta (pp)", "C-A pooled probe delta (pp)"]

    write_csv(args.output_dir / "round5_seed_results.csv", seed_headers, seed_rows)
    write_md(args.output_dir / "round5_seed_results.md", seed_headers, seed_rows)
    write_csv(args.output_dir / "round5_aggregate_results.csv", agg_headers, aggregate_rows)
    write_md(args.output_dir / "round5_aggregate_results.md", agg_headers, aggregate_rows)
    write_csv(args.output_dir / "round5_paired_deltas.csv", delta_headers, delta_rows)
    write_md(args.output_dir / "round5_paired_deltas.md", delta_headers, delta_rows)

    summary = {
        "entries_by_method": entries_by_method,
        "aggregate_by_method": aggregate_by_method,
        "paired_deltas": paired_deltas,
        "figures": {
            "main": plot_main(entries_by_method, args.output_dir),
            "probe": plot_probe(entries_by_method, args.output_dir),
        },
    }
    with (args.output_dir / "round5_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
