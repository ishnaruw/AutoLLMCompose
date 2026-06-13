#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from figure_data import (
    MODE_COLORS,
    MODE_LABELS,
    MODE_ORDER,
    AggregateResult,
    aggregate_results,
    figures_dir,
    load_all_results,
    load_pyplot,
    resolve_run_root,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate aggregate mode-summary figures from a completed run folder."
    )
    parser.add_argument("run_folder", type=Path, help="Run folder containing qXX_* result folders.")
    parser.add_argument("--output-dir", type=Path, help="Output directory. Defaults to <run_folder>/figures.")
    parser.add_argument("--png-dpi", type=int, default=300, help="PNG export resolution.")
    return parser.parse_args()


def style_axis(ax) -> None:
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#D9D9D9", linewidth=0.8)
    ax.xaxis.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def label_bars(ax, bars, values, offset=0.015, fmt="{:.3f}", fontsize=10):
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=fontsize,
        )


def label_bars_at(ax, bars, values, y_positions, fmt="{:.3f}", fontsize=10):
    for bar, value, y_position in zip(bars, values, y_positions):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y_position,
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=fontsize,
        )


def plot_average_score(plt, aggregates: list[AggregateResult], out_dir: Path, png_dpi: int) -> Path:
    fig, ax = plt.subplots(figsize=(10.5, 6.2), constrained_layout=True)
    modes = [item.mode for item in aggregates]
    means = [item.mean_score for item in aggregates]
    stds = [item.std_score for item in aggregates]
    colors = [MODE_COLORS[mode] for mode in modes]
    bars = ax.bar(modes, means, yerr=stds, capsize=6, color=colors, edgecolor="#222222", linewidth=1.0)
    ax.set_title("Average QoS-Adjusted Composition Score by Mode", pad=14)
    ax.set_xlabel("Mode")
    ax.set_ylabel("Average QoS-Adjusted Composition Score")
    label_offset = 0.018
    label_positions = [mean + std + label_offset for mean, std in zip(means, stds)]
    ymax = max(1.08, max(label_positions) + 0.04)
    ax.set_ylim(0.0, ymax)
    style_axis(ax)
    label_bars_at(ax, bars, means, label_positions, fontsize=12)
    output = out_dir / "average_composition_by_mode.png"
    fig.savefig(output, dpi=png_dpi, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_tradeoff_scatter(plt, aggregates: list[AggregateResult], out_dir: Path, png_dpi: int) -> Path:
    fig, ax = plt.subplots(figsize=(8.8, 6.8), constrained_layout=True)
    label_offsets = {
        "no_qos": (-18, -24, "right", "top"),
        "qos_pure_llm": (-24, 0, "right", "center"),
        "qos_topsis": (-18, 18, "right", "bottom"),
        "qos_hybrid": (-24, 0, "right", "center"),
    }
    for item in aggregates:
        ax.scatter(
            item.mean_functional_coverage,
            item.mean_normalized_qos,
            s=175,
            color=MODE_COLORS[item.mode],
            edgecolor="#222222",
            linewidth=1.2,
            zorder=3,
        )
        dx, dy, ha, va = label_offsets.get(item.mode, (10, 10, "left", "bottom"))
        ax.annotate(
            item.mode,
            xy=(item.mean_functional_coverage, item.mean_normalized_qos),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=12,
            fontweight="bold",
            ha=ha,
            va=va,
            clip_on=False,
        )
    ax.set_xlabel("Average Functional Coverage")
    ax.set_ylabel("Average Normalized QoS Score")
    ax.set_xlim(-0.02, 1.04)
    ax.set_ylim(-0.02, 1.04)
    ax.grid(True, color="#D9D9D9", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    output = out_dir / "coverage_qos_tradeoff.png"
    fig.savefig(output, dpi=png_dpi, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_functional_vs_qos(plt, aggregates: list[AggregateResult], out_dir: Path, png_dpi: int) -> Path:
    fig, ax = plt.subplots(figsize=(10.5, 6.0), constrained_layout=True)
    x = list(range(len(aggregates)))
    width = 0.36
    coverage = [item.mean_functional_coverage for item in aggregates]
    qos = [item.mean_normalized_qos for item in aggregates]
    coverage_bars = ax.bar(
        [pos - width / 2 for pos in x],
        coverage,
        width,
        label="Average Functional Coverage",
        color="#0B7DB4",
        edgecolor="#222222",
        linewidth=0.9,
    )
    qos_bars = ax.bar(
        [pos + width / 2 for pos in x],
        qos,
        width,
        label="Average Normalized QoS Score",
        color="#E9A300",
        edgecolor="#222222",
        linewidth=0.9,
    )
    ax.set_title("Functional Coverage and Normalized QoS by Mode", pad=14)
    ax.set_xlabel("Evaluation Mode")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels([item.mode for item in aggregates])
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False)
    style_axis(ax)
    label_bars(ax, coverage_bars, coverage, offset=0.012, fontsize=10)
    label_bars(ax, qos_bars, qos, offset=0.012, fontsize=10)
    output = out_dir / "functional_coverage_normalized_qos.png"
    fig.savefig(output, dpi=png_dpi, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_qos_components(plt, aggregates: list[AggregateResult], out_dir: Path, png_dpi: int) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(15.4, 5.3))
    fig.suptitle("QoS Component Metrics by Mode", fontsize=16, y=0.98)
    modes = [item.mode for item in aggregates]
    colors = [MODE_COLORS[mode] for mode in modes]
    panels = (
        ("Average Total Response Time", "lower is better", "seconds", [item.mean_response_time_s for item in aggregates], "{:.3f}"),
        ("Average Bottleneck Throughput", "higher is better", "kbps", [item.mean_throughput_kbps for item in aggregates], "{:.3f}"),
        ("Average Workflow Availability", "", "availability score", [item.mean_availability for item in aggregates], "{:.3f}"),
    )
    for ax, (title, subtitle, ylabel, values, fmt) in zip(axes, panels):
        bars = ax.bar(modes, values, color=colors, edgecolor="#222222", linewidth=0.8)
        ax.set_title(title, pad=24, fontsize=13)
        if subtitle:
            ax.text(0.5, 1.015, subtitle, transform=ax.transAxes, ha="center", va="bottom", color="#777777", fontsize=9)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=35)
        ymax = 1.05 if max(values) <= 1.0 else max(values) * 1.22
        ax.set_ylim(0.0, ymax)
        style_axis(ax)
        label_bars(ax, bars, values, offset=ymax * 0.015, fmt=fmt, fontsize=9)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90), w_pad=1.8)
    output = out_dir / "qos_component_metrics.png"
    fig.savefig(output, dpi=png_dpi, bbox_inches="tight")
    plt.close(fig)
    return output


def generate(run_folder: str | Path, output_dir: str | Path | None = None, png_dpi: int = 300) -> list[Path]:
    run_root = resolve_run_root(run_folder)
    rows = load_all_results(run_root)
    aggregates = aggregate_results(rows)
    out_dir = figures_dir(run_root, output_dir)
    plt = load_pyplot()
    return [
        plot_average_score(plt, aggregates, out_dir, png_dpi),
        plot_tradeoff_scatter(plt, aggregates, out_dir, png_dpi),
        plot_functional_vs_qos(plt, aggregates, out_dir, png_dpi),
        plot_qos_components(plt, aggregates, out_dir, png_dpi),
    ]


def main() -> int:
    args = parse_args()
    for output in generate(args.run_folder, args.output_dir, args.png_dpi):
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
