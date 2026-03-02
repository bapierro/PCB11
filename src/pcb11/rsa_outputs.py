"""RSA computation and plotting helpers for the ANN/MEG pipeline."""

from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np
from thingsvision.core.rsa import correlate_rdms

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def compute_rsa_timecourses(layer_rdms: list[np.ndarray], meg_rdms: np.ndarray) -> np.ndarray:
    """Compute layer-by-time RSA with ``thingsvision.core.rsa.correlate_rdms``."""

    n_images = meg_rdms.shape[0]
    if any(rdm.shape != (n_images, n_images) for rdm in layer_rdms):
        bad = [rdm.shape for rdm in layer_rdms if rdm.shape != (n_images, n_images)][0]
        raise SystemExit(f"Layer RDM shape {bad} does not match MEG shape {(n_images, n_images)}")

    n_time = meg_rdms.shape[2]
    out = np.full((len(layer_rdms), n_time), np.nan, dtype=np.float64)
    for layer_index, model_rdm in enumerate(layer_rdms):
        for time_index in range(n_time):
            rho = correlate_rdms(
                model_rdm,
                meg_rdms[:, :, time_index],
                correlation="spearman",
            )
            out[layer_index, time_index] = float(rho)
    return out


def compute_rsa_timecourses_per_subject(
    layer_rdms: list[np.ndarray],
    meg_rdms_subjects: np.ndarray,
) -> np.ndarray:
    """Compute RSA timecourses for each subject independently."""

    if meg_rdms_subjects.ndim != 4:
        raise SystemExit(
            f"Expected subject-resolved MEG RDM shape [N, N, T, S], got {meg_rdms_subjects.shape}"
        )
    n_subjects = meg_rdms_subjects.shape[3]
    n_layers = len(layer_rdms)
    n_time = meg_rdms_subjects.shape[2]
    out = np.empty((n_subjects, n_layers, n_time), dtype=np.float64)
    for subject_index in range(n_subjects):
        out[subject_index] = compute_rsa_timecourses(layer_rdms, meg_rdms_subjects[:, :, :, subject_index])
    return out


def save_rsa_outputs(
    layer_labels: Sequence[str],
    corr_matrix: np.ndarray,
    time_points: np.ndarray,
    data_dir: Path,
    plots_dir: Path,
    model_name: str,
) -> None:
    """Write RSA numeric outputs and standard visual summaries."""

    data_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    with (data_dir / "layer_index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["layer_index", "layer_name"])
        for idx, layer_name in enumerate(layer_labels):
            writer.writerow([idx, layer_name])

    with (data_dir / "layer_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["layer_index", "layer_name", "peak_corr", "peak_time_ms", "min_corr", "min_time_ms"])
        for idx, layer_name in enumerate(layer_labels):
            corr = corr_matrix[idx]
            safe = layer_name.replace(".", "_").replace("/", "_")
            np.save(data_dir / f"{safe}_spearman.npy", corr)
            np.savetxt(
                data_dir / f"{safe}_spearman.csv",
                np.column_stack((time_points, corr)),
                delimiter=",",
                header="time,corr",
                comments="",
            )
            if np.all(np.isnan(corr)):
                max_idx = 0
                min_idx = 0
            else:
                max_idx = int(np.nanargmax(corr))
                min_idx = int(np.nanargmin(corr))
            writer.writerow(
                [
                    idx,
                    layer_name,
                    float(corr[max_idx]),
                    float(time_points[max_idx]),
                    float(corr[min_idx]),
                    float(time_points[min_idx]),
                ]
            )

    finite = np.isfinite(corr_matrix)
    if np.any(finite):
        corr_vmin = float(np.nanmin(corr_matrix))
        corr_vmax = float(np.nanmax(corr_matrix))
    else:
        corr_vmin, corr_vmax = -1.0, 1.0
    if np.isclose(corr_vmin, corr_vmax):
        eps = 1e-6 if corr_vmax == 0 else abs(corr_vmax) * 0.01
        corr_vmin -= eps
        corr_vmax += eps

    fig_h, ax_h = plt.subplots(figsize=(12, min(max(6, 2 + 0.14 * len(layer_labels)), 20)))
    im = ax_h.imshow(
        corr_matrix,
        aspect="auto",
        cmap="coolwarm",
        vmin=corr_vmin,
        vmax=corr_vmax,
        interpolation="nearest",
    )
    xticks = np.linspace(0, len(time_points) - 1, 11, dtype=int)
    ax_h.set_xticks(xticks)
    ax_h.set_xticklabels([f"{time_points[idx]:.0f}" for idx in xticks])
    ytick_count = min(len(layer_labels), 20)
    yticks = np.linspace(0, len(layer_labels) - 1, ytick_count, dtype=int)
    ax_h.set_yticks(yticks)
    ax_h.set_yticklabels([str(idx) for idx in yticks])
    zero_idx = int(np.argmin(np.abs(time_points)))
    ax_h.axvline(zero_idx, color="black", linewidth=1.0, linestyle="--")
    ax_h.set_xlabel("Time (ms)")
    ax_h.set_ylabel("Layer index (see layer_index.csv)")
    ax_h.set_title(f"Layerwise MEG RSA ({model_name}, {len(layer_labels)} layers)")
    cbar = fig_h.colorbar(im, ax=ax_h, fraction=0.03, pad=0.02)
    cbar.set_label("Spearman correlation")
    fig_h.tight_layout()
    fig_h.savefig(plots_dir / "rsa_layerwise_heatmap.png", dpi=160)
    plt.close(fig_h)

    fig_l, ax_l = plt.subplots(figsize=(13.5, 6.5))
    cmap = plt.get_cmap("viridis")
    denom = max(1, len(layer_labels) - 1)
    verbose_labels = [f"{idx + 1:02d}: {layer_labels[idx]}" for idx in range(len(layer_labels))]
    max_verbose_len = max((len(label) for label in verbose_labels), default=0)
    use_compact_legend = len(layer_labels) > 8 or max_verbose_len > 28
    if use_compact_legend:
        legend_labels = [f"L{idx + 1:02d}" for idx in range(len(layer_labels))]
        legend_title = "Layers (see layer_index.csv)"
    else:
        legend_labels = verbose_labels
        legend_title = "Layers"

    for idx, corr in enumerate(corr_matrix):
        ax_l.plot(
            time_points,
            corr,
            color=cmap(idx / denom),
            alpha=0.9,
            linewidth=1.4,
            label=legend_labels[idx],
        )
    ax_l.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax_l.set_ylim(corr_vmin, corr_vmax)
    ax_l.set_xlabel("Time (ms)")
    ax_l.set_ylabel("Spearman correlation")
    ax_l.set_title(f"Layerwise RSA Time Overlay ({model_name})")
    if use_compact_legend:
        legend_columns = min(6, max(2, int(np.ceil(len(layer_labels) / 2))))
        ax_l.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            borderaxespad=0.0,
            fontsize=8,
            ncol=legend_columns,
            title=legend_title,
            title_fontsize=9,
        )
        fig_l.tight_layout(rect=(0.0, 0.14, 1.0, 1.0))
    else:
        legend_columns = 1 if len(layer_labels) <= 8 else 2
        ax_l.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            fontsize=8,
            ncol=legend_columns,
            title=legend_title,
            title_fontsize=9,
        )
        fig_l.tight_layout(rect=(0.0, 0.0, 0.74, 1.0))
    fig_l.savefig(plots_dir / "rsa_layerwise_overlay.png", dpi=160)
    plt.close(fig_l)


def smooth_data(data: np.ndarray, window: int = 20) -> np.ndarray:
    """Smooths jittery MEG data to find the true peak."""
    smoothing = np.convolve(data, np.ones(window) / window, mode="same")
    return smoothing


def save_peak_latency_plot(
    corr_by_subject: np.ndarray,
    layer_labels: Sequence[str],
    time_points: np.ndarray,
    plots_dir: Path,
    model_name: str,
    search_start_ms: float = 50.0,
    search_end_ms: float = 450.0,
    n_bootstraps: int = 1000,
) -> None:
    """Computes bootstrapped peak latencies and saves a plot."""

    n_subj, n_layers, _ = corr_by_subject.shape

    start_idx = int(np.argmin(np.abs(time_points - search_start_ms)))
    end_idx = int(np.argmin(np.abs(time_points - search_end_ms)))

    final_peaks: list[float] = []
    final_errors: list[float] = []

    for layer_idx in range(n_layers):
        layer_data = corr_by_subject[:, layer_idx, :]  # Shape: (Subjects, Time)

        bootstrapped_peaks: list[float] = []
        for _ in range(n_bootstraps):
            sample_indices = np.random.choice(n_subj, size=n_subj, replace=True)
            boot_sample = layer_data[sample_indices, :]  # Shape: (Subjects, Time)
            boot_mean = boot_sample.mean(axis=0)  # Shape: (Time,)
            boot_smoothed = smooth_data(boot_mean)

            window_segment = boot_smoothed[start_idx:end_idx]
            peak_idx = start_idx + int(np.argmax(window_segment))
            bootstrapped_peaks.append(float(time_points[peak_idx]))

        final_peaks.append(float(np.mean(bootstrapped_peaks)))
        final_errors.append(float(np.std(bootstrapped_peaks)))

    layer_indices = list(range(1, len(layer_labels) + 1))
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(layer_labels)))

    ax.plot(layer_indices, final_peaks, "-", color="black", linewidth=1.5, zorder=1)

    for i in range(n_layers):
        if not np.isnan(final_peaks[i]):
            ax.errorbar(
                i + 1,
                final_peaks[i],
                yerr=final_errors[i],
                fmt="o",
                color=colors[i],
                markersize=10,
                capsize=5,
                capthick=2,
                linewidth=2,
                zorder=2,
            )

    ax.set_title(f"Peak Latency per Layer ({model_name}) (Bootstrapped SEM)", fontsize=14)
    ax.set_xticks(layer_indices)
    ax.set_xticklabels(layer_labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Time of Peak Match (ms)")
    ax.set_xlabel("Layers")
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    fig.tight_layout()
    fig.savefig(plots_dir / "rsa_peak_latency.png", dpi=160)
    plt.close(fig)


def compute_model_rsa(layer_rdms: list[np.ndarray], model_rdm: np.ndarray) -> np.ndarray:
    """Compute RSA between each ANN layer and a static Model RDM."""
    n_images = model_rdm.shape[0]
    out = np.empty(len(layer_rdms), dtype=np.float64)
    for idx, layer_rdm in enumerate(layer_rdms):
        if layer_rdm.shape != (n_images, n_images):
            raise SystemExit(f"Layer RDM shape {layer_rdm.shape} does not match Model RDM shape {(n_images, n_images)}")
        rho = correlate_rdms(model_rdm, layer_rdm, correlation="spearman")
        out[idx] = float(rho)
    return out


def save_spatial_vs_semantic_plot(
    spatial_corr: np.ndarray,
    semantic_corr: np.ndarray,
    layer_labels: Sequence[str],
    plots_dir: Path,
    model_name: str,
) -> None:
    """Plot spatial vs semantic RSA across ANN layers."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    layer_indices = list(range(1, len(layer_labels) + 1))
    
    ax.plot(layer_indices, spatial_corr, "-o", color="#d62728", linewidth=2, label="Spatial Structure", zorder=2)
    ax.plot(layer_indices, semantic_corr, "-s", color="#1f77b4", linewidth=2, label="Semantic Content", zorder=2)
    
    ax.set_title(f"Spatial vs Semantic RSA - {model_name}", fontsize=14)
    ax.set_xticks(layer_indices)
    ax.set_xticklabels(layer_labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Spearman Correlation")
    ax.set_xlabel("Layers")
    ax.legend(loc="best")
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    
    fig.tight_layout()
    fig.savefig(plots_dir / "rsa_spatial_vs_semantic.png", dpi=160)
    plt.close(fig)

