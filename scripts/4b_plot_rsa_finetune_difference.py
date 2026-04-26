#!/usr/bin/env python3
import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "outputs" / ".matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASELINE_LAYERS = {
    "alexnet": [
        "features.2",
        "features.5",
        "features.7",
        "features.9",
        "features.12",
        "classifier.2",
        "classifier.5",
        "classifier.6",
    ],
    "resnet50": ["layer1", "layer2", "layer3", "layer4"],
}

FINETUNED_LAYERS = {
    "finetuned_all": {
        "alexnet": [
            "features_2",
            "features_5",
            "features_7",
            "features_9",
            "features_12",
            "shared_classifier_2",
            "shared_classifier_5",
            "head_app",
            "head_sem",
            "head_str",
        ],
        "resnet50": ["layer1", "layer2", "layer3", "layer4", "head_app", "head_sem", "head_str"],
    },
    "finetuned_appearance": {
        "alexnet": [
            "features_2",
            "features_5",
            "features_7",
            "features_9",
            "features_12",
            "classifier_2",
            "classifier_5",
            "classifier_6",
        ],
        "resnet50": ["layer1", "layer2", "layer3", "layer4"],
    },
    "finetuned_semantic": {
        "alexnet": [
            "features_2",
            "features_5",
            "features_7",
            "features_9",
            "features_12",
            "classifier_2",
            "classifier_5",
            "classifier_6",
        ],
        "resnet50": ["layer1", "layer2", "layer3", "layer4"],
    },
    "finetuned_structure": {
        "alexnet": [
            "features_2",
            "features_5",
            "features_7",
            "features_9",
            "features_12",
            "classifier_2",
            "classifier_5",
            "classifier_6",
        ],
        "resnet50": ["layer1", "layer2", "layer3", "layer4"],
    },
}


def safe_layer_name(layer_name):
    return layer_name.replace(".", "_").replace("/", "_")


def comparable_layers(model_name, finetuned_pipeline):
    baseline_layers = BASELINE_LAYERS[model_name]
    finetuned_layers = FINETUNED_LAYERS[finetuned_pipeline][model_name]

    if finetuned_pipeline == "finetuned_all" and model_name == "alexnet":
        mapping = {
            "features.2": "features_2",
            "features.5": "features_5",
            "features.7": "features_7",
            "features.9": "features_9",
            "features.12": "features_12",
            "classifier.2": "shared_classifier_2",
            "classifier.5": "shared_classifier_5",
        }
        labels = {
            "features.2": "features 2",
            "features.5": "features 5",
            "features.7": "features 7",
            "features.9": "features 9",
            "features.12": "features 12",
            "classifier.2": "classifier 2",
            "classifier.5": "classifier 5",
        }
        return [(base, fine, labels[base]) for base, fine in mapping.items()]

    finetuned_by_safe_name = {safe_layer_name(layer): layer for layer in finetuned_layers}
    pairs = []
    for baseline_layer in baseline_layers:
        safe_name = safe_layer_name(baseline_layer)
        if safe_name in finetuned_by_safe_name:
            pairs.append((baseline_layer, finetuned_by_safe_name[safe_name], safe_name))
    return pairs


def load_rsa_timecourse(rsa_dir, layer_name):
    safe_name = safe_layer_name(layer_name)
    mean_file = rsa_dir / f"{safe_name}_rsa_spearman_mean.npy"
    single_file = rsa_dir / f"{safe_name}_rsa_spearman.npy"

    if mean_file.exists():
        return np.load(mean_file)
    if single_file.exists():
        return np.load(single_file)
    raise FileNotFoundError(f"Missing RSA data for {layer_name}: expected {mean_file.name} or {single_file.name}")


def layer_summary(layer_label, baseline_data, finetuned_data, time_points, search_mask):
    baseline_window = baseline_data[search_mask]
    finetuned_window = finetuned_data[search_mask]
    time_window = time_points[search_mask]
    diff = finetuned_data - baseline_data
    diff_window = diff[search_mask]

    baseline_peak_idx = int(np.nanargmax(baseline_window))
    finetuned_peak_idx = int(np.nanargmax(finetuned_window))
    max_diff_idx = int(np.nanargmax(diff_window))

    baseline_peak = float(baseline_window[baseline_peak_idx])
    finetuned_peak = float(finetuned_window[finetuned_peak_idx])
    max_diff = float(diff_window[max_diff_idx])

    return {
        "layer": layer_label,
        "baseline_peak_rsa": baseline_peak,
        "baseline_peak_time_ms": float(time_window[baseline_peak_idx]),
        "finetuned_peak_rsa": finetuned_peak,
        "finetuned_peak_time_ms": float(time_window[finetuned_peak_idx]),
        "peak_rsa_difference": finetuned_peak - baseline_peak,
        "max_timepoint_difference": max_diff,
        "max_timepoint_difference_ms": float(time_window[max_diff_idx]),
    }


def save_summary_csv(rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "layer",
        "baseline_peak_rsa",
        "baseline_peak_time_ms",
        "finetuned_peak_rsa",
        "finetuned_peak_time_ms",
        "peak_rsa_difference",
        "max_timepoint_difference",
        "max_timepoint_difference_ms",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_difference(time_points, layer_labels, difference_curves, peak_differences, output_path, title):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_time, ax_peak) = plt.subplots(
        2,
        1,
        figsize=(12, 10),
        gridspec_kw={"height_ratios": [2, 1]},
        constrained_layout=True,
    )
    colors = plt.cm.viridis(np.linspace(0, 1, len(layer_labels)))

    for layer_label, difference, color in zip(layer_labels, difference_curves, colors):
        ax_time.plot(time_points, difference, label=layer_label, color=color, linewidth=2)

    ax_time.axhline(0, color="black", linewidth=1)
    ax_time.axvline(0, color="black", linestyle="--", linewidth=1)
    ax_time.set_title(title)
    ax_time.set_ylabel("RSA difference (fine-tuned - baseline)")
    ax_time.set_xlabel("Time from stimulus onset (ms)")
    ax_time.set_xlim(float(time_points[0]), float(time_points[-1]))
    ax_time.grid(alpha=0.2)
    ax_time.legend(loc="upper right", fontsize="small", ncol=2)

    bar_colors = ["#2ca25f" if diff >= 0 else "#de2d26" for diff in peak_differences]
    x = np.arange(len(layer_labels))
    ax_peak.bar(x, peak_differences, color=bar_colors)
    ax_peak.axhline(0, color="black", linewidth=1)
    ax_peak.set_title("Peak RSA height difference by layer")
    ax_peak.set_ylabel("Peak RSA difference")
    ax_peak.set_xticks(x)
    ax_peak.set_xticklabels(layer_labels, rotation=35, ha="right")
    ax_peak.grid(axis="y", alpha=0.2)

    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare layerwise RSA from a fine-tuned model against the pretrained baseline."
    )
    parser.add_argument("--model", choices=sorted(BASELINE_LAYERS), default="alexnet")
    parser.add_argument(
        "--finetuned-pipeline",
        choices=sorted(FINETUNED_LAYERS),
        default="finetuned_all",
        help="Fine-tuned output folder to compare against outputs/clean_baseline.",
    )
    parser.add_argument("--baseline-pipeline", default="clean_baseline")
    parser.add_argument("--time-start-ms", type=float, default=-200.0)
    parser.add_argument("--time-end-ms", type=float, default=1000.0)
    parser.add_argument("--peak-start-ms", type=float, default=-200.0)
    parser.add_argument("--peak-end-ms", type=float, default=1000.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to outputs/rsa_finetune_comparison/<model>_<fine_pipeline>.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    baseline_dir = PROJECT_ROOT / "outputs" / args.baseline_pipeline / "rsa" / args.model
    finetuned_dir = PROJECT_ROOT / "outputs" / args.finetuned_pipeline / "rsa" / args.model
    output_dir = args.output_dir or PROJECT_ROOT / "outputs" / "rsa_finetune_comparison" / (
        f"{args.model}_{args.finetuned_pipeline}"
    )

    pairs = comparable_layers(args.model, args.finetuned_pipeline)
    if not pairs:
        raise ValueError(f"No comparable layers found for {args.model} and {args.finetuned_pipeline}.")

    missing_dirs = [path for path in (baseline_dir, finetuned_dir) if not path.exists()]
    if missing_dirs:
        formatted = "\n".join(f" - {path}" for path in missing_dirs)
        raise FileNotFoundError(f"Missing RSA output directories:\n{formatted}")

    rows = []
    layer_labels = []
    difference_curves = []
    peak_differences = []
    expected_n_time = None

    for baseline_layer, finetuned_layer, layer_label in pairs:
        baseline_data = np.asarray(load_rsa_timecourse(baseline_dir, baseline_layer), dtype=float).squeeze()
        finetuned_data = np.asarray(load_rsa_timecourse(finetuned_dir, finetuned_layer), dtype=float).squeeze()
        if baseline_data.shape != finetuned_data.shape:
            raise ValueError(
                f"Shape mismatch for {layer_label}: baseline {baseline_data.shape}, "
                f"fine-tuned {finetuned_data.shape}"
            )
        if baseline_data.ndim != 1:
            raise ValueError(f"Expected 1D RSA timecourse for {layer_label}, got shape {baseline_data.shape}.")
        if expected_n_time is None:
            expected_n_time = baseline_data.shape[0]
        elif baseline_data.shape[0] != expected_n_time:
            raise ValueError(f"Unexpected number of timepoints for {layer_label}.")

        time_points = np.linspace(args.time_start_ms, args.time_end_ms, baseline_data.shape[0])
        search_mask = (time_points >= args.peak_start_ms) & (time_points <= args.peak_end_ms)
        if not np.any(search_mask):
            raise ValueError("Peak search window does not overlap the RSA time axis.")

        summary = layer_summary(layer_label, baseline_data, finetuned_data, time_points, search_mask)
        rows.append(summary)
        layer_labels.append(layer_label)
        difference_curves.append(finetuned_data - baseline_data)
        peak_differences.append(summary["peak_rsa_difference"])

    plot_path = output_dir / "rsa_finetuned_minus_baseline.png"
    csv_path = output_dir / "rsa_finetuned_minus_baseline_summary.csv"
    title = (
        f"{args.model.upper()} RSA Difference: "
        f"{args.finetuned_pipeline.replace('_', ' ').title()} - Pretrained Baseline"
    )
    plot_difference(time_points, layer_labels, difference_curves, peak_differences, plot_path, title)
    save_summary_csv(rows, csv_path)

    print(f"Saved plot to: {plot_path}")
    print(f"Saved summary to: {csv_path}")
    print("\nLayers with higher fine-tuned peak RSA:")
    higher_layers = [row for row in rows if row["peak_rsa_difference"] > 0]
    if higher_layers:
        for row in higher_layers:
            print(
                f" - {row['layer']}: +{row['peak_rsa_difference']:.4f} "
                f"(fine-tuned peak {row['finetuned_peak_rsa']:.4f}, "
                f"baseline peak {row['baseline_peak_rsa']:.4f})"
            )
    else:
        print(" - None")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
