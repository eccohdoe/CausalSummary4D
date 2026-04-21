from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent
STYLE_PATH = Path("/home/xiaoql26/.codex/skills/deepscientist-figure-polish/assets/deepscientist-academic.mplstyle")


METHOD_ORDER = [
    ("causalonly", "Causal-only", "none"),
    ("A", "A", "summary delta"),
    ("C", "C", "projected pooled-state delta"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build round-4 matched A/C paper tables and figures.")
    parser.add_argument("--a-dir", required=True, type=Path)
    parser.add_argument("--c-dir", required=True, type=Path)
    parser.add_argument("--causal-dir", type=Path, default=None)
    parser.add_argument("--a-probe", required=True, type=Path)
    parser.add_argument("--c-probe", required=True, type=Path)
    parser.add_argument("--causal-probe", type=Path, default=None)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_run(run_dir: Optional[Path]) -> Optional[Dict]:
    if run_dir is None:
        return None
    return {
        "run_dir": str(run_dir),
        "summary": load_json(run_dir / "summary.json"),
        "history": load_json(run_dir / "history.json"),
        "resolved_config": load_json(run_dir / "resolved_config.yaml") if False else None,
    }


def load_probe(probe_path: Optional[Path]) -> Optional[Dict]:
    if probe_path is None:
        return None
    return load_json(probe_path)


def best_history_row(run: Dict) -> Dict:
    summary = run["summary"]
    history = run["history"]
    best_epoch = summary.get("best_epoch")
    if best_epoch is None:
        best_epoch = max(range(len(history)), key=lambda idx: history[idx].get("val_video_acc1", float("-inf")))
    return history[best_epoch]


def maybe(value, ndigits: int = 4):
    if value is None:
        return "-"
    if isinstance(value, str):
        return value
    return round(float(value), ndigits)


def percent(value: Optional[float], ndigits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{100.0 * float(value):.{ndigits}f}"


def render_markdown_table(headers: List[str], rows: List[List[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines) + "\n"


def write_csv(path: Path, headers: List[str], rows: List[List[object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def method_color(method_key: str) -> str:
    return {
        "causalonly": "#B7A99A",
        "A": "#8A9199",
        "C": "#7F8F84",
    }[method_key]


def build_rows(run_map: Dict[str, Optional[Dict]], probe_map: Dict[str, Optional[Dict]]):
    core_rows = []
    mech_rows = []
    summary_payload = {"methods": {}}

    for method_key, method_name, pred_name in METHOD_ORDER:
        run = run_map.get(method_key)
        if run is None:
            continue
        summary = run["summary"]
        best_row = best_history_row(run)
        probe = probe_map.get(method_key)

        summary_probe = None
        pooled_probe = None
        summary_delta = None
        pooled_delta = None
        proj_delta = None
        if probe is not None:
            pooled_probe = probe["classification"]["pooled_final"]["mlp"]["best_val_video_acc"]
            if "summary_final" in probe["classification"]:
                summary_probe = probe["classification"]["summary_final"]["mlp"]["best_val_video_acc"]
            pooled_delta = probe["delta_regression"]["pooled_state"]["mlp"]["best_val_mse"]
            if "summary" in probe["delta_regression"]:
                summary_delta = probe["delta_regression"]["summary"]["mlp"]["best_val_mse"]
            if "projected_pooled_target" in probe["delta_regression"]:
                proj_delta = probe["delta_regression"]["projected_pooled_target"]["mlp"]["best_val_mse"]

        core_rows.append([
            method_name,
            pred_name,
            percent(summary.get("best_val_video_acc1")),
            percent(summary.get("final_val_video_acc1")),
            summary.get("best_epoch", "-"),
            "pass" if summary.get("causality", {}).get("passed") else "n/a",
            percent(summary_probe) if summary_probe is not None else "-",
            percent(pooled_probe) if pooled_probe is not None else "-",
        ])

        mech_rows.append([
            method_name,
            maybe(summary_delta),
            maybe(pooled_delta),
            maybe(proj_delta),
            maybe(best_row.get("summary_std")),
            maybe(best_row.get("summary_std_min")),
            maybe(best_row.get("slot_corr")),
            maybe(best_row.get("slot_max_corr")),
        ])

        summary_payload["methods"][method_key] = {
            "method_name": method_name,
            "predictive_target": pred_name,
            "best_val_video_acc1": summary.get("best_val_video_acc1"),
            "final_val_video_acc1": summary.get("final_val_video_acc1"),
            "best_epoch": summary.get("best_epoch"),
            "causality": summary.get("causality"),
            "summary_probe_acc": summary_probe,
            "pooled_probe_acc": pooled_probe,
            "summary_delta_mse": summary_delta,
            "pooled_delta_mse": pooled_delta,
            "projected_pooled_target_delta_mse": proj_delta,
            "summary_std": best_row.get("summary_std"),
            "summary_std_min": best_row.get("summary_std_min"),
            "slot_corr": best_row.get("slot_corr"),
            "slot_max_corr": best_row.get("slot_max_corr"),
            "history_path": str(Path(run["run_dir"]) / "history.json"),
            "probe_path": str(probe_map[method_key]) if probe_map.get(method_key) is not None else None,
        }

    return core_rows, mech_rows, summary_payload


def plot_main_results(run_map: Dict[str, Optional[Dict]], output_dir: Path) -> List[str]:
    plt.style.use(str(STYLE_PATH))
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    keys = [key for key, _, _ in METHOD_ORDER if run_map.get(key) is not None]
    labels = [name for key, name, _ in METHOD_ORDER if run_map.get(key) is not None]
    values = [100.0 * run_map[key]["summary"]["best_val_video_acc1"] for key in keys]
    colors = [method_color(key) for key in keys]
    bars = ax.bar(labels, values, color=colors, width=0.62)
    ax.set_ylabel("Best val video acc (%)")
    ax.set_ylim(74, max(values) + 2.5 if values else 85)
    ax.set_title("Matched full-data MSR results")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2.0, value + 0.15, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    png = output_dir / "round4_main_results.png"
    pdf = output_dir / "round4_main_results.pdf"
    fig.savefig(png, dpi=240, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [str(png), str(pdf)]


def plot_probe_comparison(probe_map: Dict[str, Optional[Dict]], output_dir: Path) -> List[str]:
    plt.style.use(str(STYLE_PATH))
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    methods = ["A", "C"]
    labels = [m for m in methods if probe_map.get(m) is not None]
    x = np.arange(len(labels))
    width = 0.32
    pooled = [100.0 * probe_map[m]["classification"]["pooled_final"]["mlp"]["best_val_video_acc"] for m in labels]
    summary = [100.0 * probe_map[m]["classification"]["summary_final"]["mlp"]["best_val_video_acc"] for m in labels]
    ax.bar(x - width / 2.0, pooled, width=width, color="#B7A99A", label="Pooled probe")
    ax.bar(x + width / 2.0, summary, width=width, color="#7F8F84", label="Summary probe")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Video probe acc (%)")
    ax.set_ylim(35, max(pooled + summary) + 3.0)
    ax.set_title("Representation discriminativeness")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    png = output_dir / "round4_probe_comparison.png"
    pdf = output_dir / "round4_probe_comparison.pdf"
    fig.savefig(png, dpi=240, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [str(png), str(pdf)]


def plot_training_curves(run_map: Dict[str, Optional[Dict]], output_dir: Path) -> List[str]:
    plt.style.use(str(STYLE_PATH))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for method_key, label, _ in METHOD_ORDER:
        run = run_map.get(method_key)
        if run is None:
            continue
        history = run["history"]
        epochs = [row["epoch"] for row in history]
        val_acc = [100.0 * row.get("val_video_acc1", np.nan) for row in history]
        axes[0].plot(epochs, val_acc, marker="o", markersize=3, linewidth=1.8, color=method_color(method_key), label=label)
        if method_key != "causalonly":
            loss_pred = [row.get("loss_pred", np.nan) for row in history]
            axes[1].plot(epochs, loss_pred, marker="o", markersize=3, linewidth=1.8, color=method_color(method_key), label=label)

    axes[0].set_title("Validation accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Val video acc (%)")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, loc="lower right")

    axes[1].set_title("Predictive loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("loss_pred")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, loc="upper right")

    fig.tight_layout()
    png = output_dir / "round4_training_curves.png"
    pdf = output_dir / "round4_training_curves.pdf"
    fig.savefig(png, dpi=240, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [str(png), str(pdf)]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    run_map = {
        "A": load_run(args.a_dir),
        "C": load_run(args.c_dir),
        "causalonly": load_run(args.causal_dir),
    }
    probe_map = {
        "A": load_probe(args.a_probe),
        "C": load_probe(args.c_probe),
        "causalonly": load_probe(args.causal_probe),
    }

    core_rows, mech_rows, summary_payload = build_rows(run_map, probe_map)

    core_headers = [
        "Method",
        "Predictive target",
        "Best val video acc",
        "Final val video acc",
        "Best epoch",
        "Causality",
        "Summary probe acc",
        "Pooled probe acc",
    ]
    mech_headers = [
        "Method",
        "Summary delta MSE",
        "Pooled delta MSE",
        "Projected pooled-target delta MSE",
        "summary_std",
        "summary_std_min",
        "slot_corr",
        "slot_max_corr",
    ]

    core_md = render_markdown_table(core_headers, core_rows)
    mech_md = render_markdown_table(mech_headers, mech_rows)
    (args.output_dir / "core_results.md").write_text(core_md, encoding="utf-8")
    (args.output_dir / "mechanism_results.md").write_text(mech_md, encoding="utf-8")
    write_csv(args.output_dir / "core_results.csv", core_headers, core_rows)
    write_csv(args.output_dir / "mechanism_results.csv", mech_headers, mech_rows)

    figure_exports = {
        "main_results": plot_main_results(run_map, args.output_dir),
        "probe_comparison": plot_probe_comparison(probe_map, args.output_dir),
        "training_curves": plot_training_curves(run_map, args.output_dir),
    }
    summary_payload["figures"] = figure_exports
    summary_payload["tables"] = {
        "core_markdown": str(args.output_dir / "core_results.md"),
        "core_csv": str(args.output_dir / "core_results.csv"),
        "mechanism_markdown": str(args.output_dir / "mechanism_results.md"),
        "mechanism_csv": str(args.output_dir / "mechanism_results.csv"),
    }
    (args.output_dir / "round4_summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
