# -*- coding: utf-8 -*-
"""
compare_selected_agents.py

Training-based comparison of the selected agents:
- U1 Seed 3
- U2 Seed 2

Creates six comparison figures, CSV summaries, and rendered tables.
Transparent lines are logged TensorBoard values. Solid lines are exponentially
smoothed for display only (weight = 0.60).
"""

from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, MultipleLocator
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ======================================================
# CONFIGURATION
# ======================================================
MAX_STEP = 500_000
SMOOTHING_WEIGHT = 0.60
PLOT_MODE = "thesis"  # change to "presentation" if needed

OUTPUT_DIR = Path(
    r"C:\Users\49151\Desktop\Master\07_Masterthesis"
    r"\00_InTas_New - RL\Pyscripts\selected_agent_comparison"
)

RUNS = {
    "U1 Seed 3": {
        "variant": "U1",
        "seed": 3,
        "color": "#0072B2",
        "file": Path(
            r"C:\Users\49151\Desktop\Master\07_Masterthesis"
            r"\00_InTas_New - RL\Pyscripts\logs\U1\seed_3"
            r"\20260814_001010\tensorboard\PPO_1"
            r"\events.out.tfevents.1786659015.LAPTOP-JE5U4U1C.14128.1"
        ),
    },
    "U2 Seed 2": {
        "variant": "U2",
        "seed": 2,
        "color": "#E69F00",
        "file": Path(
            r"C:\Users\49151\Desktop\Master\07_Masterthesis"
            r"\00_InTas_New - RL\Pyscripts\logs\U2\seed_2"
            r"\20260812_171310\tensorboard\PPO_1"
            r"\events.out.tfevents.1786547596.LAPTOP-JE5U4U1C.12840.0"
        ),
    },
}

if PLOT_MODE == "presentation":
    FIG_SINGLE = (10.5, 6.0)
    FIG_DOUBLE = (10.5, 9.5)
    FIG_TRIPLE = (10.5, 12.0)
    TITLE_SIZE, LABEL_SIZE, TICK_SIZE, LEGEND_SIZE = 18, 15, 13, 12
else:
    FIG_SINGLE = (8.0, 4.6)
    FIG_DOUBLE = (8.0, 8.0)
    FIG_TRIPLE = (8.0, 10.0)
    TITLE_SIZE, LABEL_SIZE, TICK_SIZE, LEGEND_SIZE = 14, 12, 10, 10

plt.rcParams.update({
    "axes.titlesize": TITLE_SIZE,
    "axes.labelsize": LABEL_SIZE,
    "xtick.labelsize": TICK_SIZE,
    "ytick.labelsize": TICK_SIZE,
    "legend.fontsize": LEGEND_SIZE,
})

# ======================================================
# DATA HELPERS
# ======================================================
_CACHE = {}


def get_accumulator(path):
    key = str(path)
    if key not in _CACHE:
        if not path.is_file():
            raise FileNotFoundError(f"Event file not found:\n{path}")
        acc = EventAccumulator(str(path), size_guidance={"scalars": 0})
        acc.Reload()
        _CACHE[key] = acc
    return _CACHE[key]


def load_scalar(run_name, tag):
    path = RUNS[run_name]["file"]
    acc = get_accumulator(path)
    if tag not in acc.Tags().get("scalars", []):
        raise KeyError(f"Tag '{tag}' not found for {run_name}")
    events = acc.Scalars(tag)
    steps = np.asarray([e.step for e in events], dtype=np.int64)
    values = np.asarray([e.value for e in events], dtype=np.float64)
    mask = (steps <= MAX_STEP) & np.isfinite(values)
    return steps[mask], values[mask]


def smooth(values, weight=SMOOTHING_WEIGHT):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return values.copy()
    result = np.empty_like(values)
    result[0] = values[0]
    for index in range(1, len(values)):
        result[index] = weight * result[index - 1] + (1.0 - weight) * values[index]
    return result


def kfmt(value, _position):
    return "0" if value == 0 else f"{int(value / 1000)}k"


def pctfmt(value, _position):
    return f"{value:.0f}%"


def setup_axis(axis, ylabel, ylim=None):
    axis.set_xlim(0, MAX_STEP)
    if ylim is not None:
        axis.set_ylim(*ylim)
    axis.set_xlabel("Training timesteps")
    axis.set_ylabel(ylabel)
    axis.xaxis.set_major_locator(MultipleLocator(100_000))
    axis.xaxis.set_major_formatter(FuncFormatter(kfmt))
    axis.grid(True, alpha=0.25)


def plot_run(
    axis,
    run_name,
    tag,
    transform=None,
    show_end_value=False,
    end_value_format=".3f",
):

    steps, values = load_scalar(
        run_name,
        tag
    )

    if transform is not None:
        values = transform(values)

    smoothed = smooth(values)

    color = RUNS[run_name]["color"]

    final_value = float(
        values[-1]
    )

    if show_end_value:

        if end_value_format == "percent":

            label = (
                f"{run_name} "
                f"(end: {final_value:.1f}%)"
            )

        else:

            label = (
                f"{run_name} "
                f"(end: "
                f"{format(final_value, end_value_format)})"
            )

    else:

        label = run_name

    axis.plot(
        steps,
        values,
        color=color,
        linewidth=0.8,
        alpha=0.12,
        zorder=1,
    )

    axis.plot(
        steps,
        smoothed,
        color=color,
        linewidth=2.3,
        label=label,
        zorder=3,
    )

    return (
        steps,
        values,
        smoothed
    )


def save_figure(figure, filename):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    for extension, kwargs in {"png": {"dpi": 300}, "svg": {}, "pdf": {}}.items():
        path = OUTPUT_DIR / f"{filename}.{extension}"
        figure.savefig(path, bbox_inches="tight", **kwargs)
        print(f"Saved: {path}")
    plt.close(figure)


def save_csv(rows, filename):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / filename
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path}")


def render_table(rows, columns, labels, title, filename, formats=None):
    formats = formats or {}
    cell_text = []
    for row in rows:
        cell_text.append([
            formats.get(column, str)(row[column]) if column in formats else str(row[column])
            for column in columns
        ])

    fig, axis = plt.subplots(figsize=(max(8.0, 1.5 * len(columns)), 1.8 + 0.5 * len(rows)))
    axis.axis("off")
    table = axis.table(
        cellText=cell_text,
        colLabels=labels,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10 if PLOT_MODE == "thesis" else 12)
    table.scale(1.0, 1.45)
    for (row_index, _column_index), cell in table.get_celld().items():
        cell.set_edgecolor("#B0B0B0")
        if row_index == 0:
            cell.set_facecolor("#E8EEF7")
            cell.set_text_props(weight="bold")
        elif row_index % 2 == 0:
            cell.set_facecolor("#F7F7F7")
    axis.set_title(title, fontsize=TITLE_SIZE, pad=12)
    save_figure(fig, filename)

# ======================================================
# 1. REWARD CONVERGENCE, SEPARATE SUBPLOTS
# ======================================================
def create_reward_convergence():
    figure, axes = plt.subplots(2, 1, figsize=FIG_DOUBLE, sharex=True)
    settings = [
        ("U1 Seed 3", (0, 330)),
        ("U2 Seed 2", (0, 180)),
    ]
    rows = []
    for axis, (run_name, ylim) in zip(axes, settings):
        steps, values, smoothed = plot_run(
            axis, run_name, "rollout/ep_rew_mean", show_end_value=True
        )
        setup_axis(axis, "Mean episodic reward", ylim)
        axis.set_title(f"{run_name}: Mean Episodic Reward")
        axis.legend(loc="lower right")
        rows.append({
            "Agent": run_name,
            "Initial logged reward": float(values[0]),
            "Final logged reward": float(values[-1]),
            "Final smoothed reward": float(smoothed[-1]),
            "Minimum reward": float(np.min(values)),
            "Maximum reward": float(np.max(values)),
        })
    save_figure(figure, "comparison_01_reward_convergence")
    save_csv(rows, "comparison_table_01_reward_summary.csv")
    render_table(
        rows,
        ["Agent", "Initial logged reward", "Final logged reward", "Final smoothed reward", "Minimum reward", "Maximum reward"],
        ["Agent", "Initial", "Final", "Final smooth", "Min", "Max"],
        "Selected-Agent Reward Summary",
        "comparison_table_01_reward_summary",
        formats={key: (lambda value: f"{value:.2f}") for key in [
            "Initial logged reward", "Final logged reward", "Final smoothed reward", "Minimum reward", "Maximum reward"
        ]},
    )

# ======================================================
# 2. DIRECT INTERACTION RATE COMPARISON
# ======================================================
def create_interaction_comparison():
    figure, axis = plt.subplots(figsize=FIG_SINGLE)
    rows = []
    for run_name in RUNS:
        steps, values = load_scalar(run_name, "interaction/rate")
        values_percent = values * 100.0
        smoothed = smooth(values_percent)
        color = RUNS[run_name]["color"]
        mean_rate = float(np.mean(values_percent))
        median_rate = float(np.median(values_percent))
        maximum_rate = float(np.max(values_percent))

        axis.plot(steps, values_percent, color=color, linewidth=0.8, alpha=0.12)
        axis.plot(steps, smoothed, color=color, linewidth=2.3,
                  label=f"{run_name} (mean: {mean_rate:.1f}%)")
        axis.axhline(mean_rate, color=color, linewidth=1.2, linestyle=":", alpha=0.9)
        rows.append({
            "Agent": run_name,
            "Mean local rate (%)": mean_rate,
            "Median local rate (%)": median_rate,
            "Maximum local rate (%)": maximum_rate,
            "Logged windows": len(values_percent),
        })

    setup_axis(axis, "Interaction rate", (0, 35))
    axis.yaxis.set_major_locator(MultipleLocator(5))
    axis.yaxis.set_major_formatter(FuncFormatter(pctfmt))
    axis.set_title("Selected Agents: Local Interaction Rate")
    axis.legend(loc="upper right")
    save_figure(figure, "comparison_02_interaction_rate")
    save_csv(rows, "comparison_table_02_interaction_rate.csv")
    render_table(
        rows,
        ["Agent", "Mean local rate (%)", "Median local rate (%)", "Maximum local rate (%)", "Logged windows"],
        ["Agent", "Mean (%)", "Median (%)", "Maximum (%)", "Windows"],
        "Selected-Agent Local Interaction Rates",
        "comparison_table_02_interaction_rate",
        formats={
            "Mean local rate (%)": lambda value: f"{value:.2f}",
            "Median local rate (%)": lambda value: f"{value:.2f}",
            "Maximum local rate (%)": lambda value: f"{value:.2f}",
        },
    )

# ======================================================
# 3. STF COMPONENTS
# ======================================================
def create_stf_comparison():
    settings = [
        ("raw/S", "Safety score", (0, 1.2)),
        ("raw/T", "Throughput proxy", (0, 0.45)),
        ("raw/F", "Fairness score", (0, 1)),
    ]
    figure, axes = plt.subplots(3, 1, figsize=FIG_TRIPLE, sharex=True)
    rows = []
    for axis, (tag, title, ylim) in zip(axes, settings):
        for run_name in RUNS:
            plot_run(axis, run_name, tag, show_end_value=True)
        setup_axis(axis, title, ylim)
        axis.set_title(title)
        axis.legend(loc="best")

    for run_name in RUNS:
        row = {"Agent": run_name}
        for tag, label in [("raw/S", "Safety"), ("raw/T", "Throughput"), ("raw/F", "Fairness")]:
            _, values = load_scalar(run_name, tag)
            row[f"Final {label}"] = float(values[-1])
            row[f"Mean {label}"] = float(np.mean(values))
            row[f"Minimum {label}"] = float(np.min(values))
            row[f"Maximum {label}"] = float(np.max(values))
        rows.append(row)

    save_figure(figure, "comparison_03_stf_components")
    save_csv(rows, "comparison_table_03_stf_components.csv")
    compact_columns = ["Agent", "Final Safety", "Final Throughput", "Final Fairness", "Mean Safety", "Mean Throughput", "Mean Fairness"]
    render_table(
        rows,
        compact_columns,
        ["Agent", "Final S", "Final T", "Final F", "Mean S", "Mean T", "Mean F"],
        "Selected-Agent STF Components",
        "comparison_table_03_stf_components",
        formats={column: (lambda value: f"{value:.3f}") for column in compact_columns if column != "Agent"},
    )

# ======================================================
# 4. MEAN IMPORTANCE, DESCRIPTIVE COMPARISON
# ======================================================
def create_importance_comparison():
    settings = [
        ("importance_mean/S", "Mean Safety Importance", (0, 1)),
        ("importance_mean/T", "Mean Throughput Importance", (0, 1)),
        ("importance_mean/F", "Mean Fairness Importance", (0, 1)),
    ]
    figure, axes = plt.subplots(4, 1, figsize=(8.0, 12.0), sharex=True)
    rows = []

    for axis, (tag, title, ylim) in zip(axes[:3], settings):
        for run_name in RUNS:
            plot_run(axis, run_name, tag, show_end_value=True)
        setup_axis(axis, "Relative contribution", ylim)
        axis.set_title(title)
        axis.legend(loc="best")

    ethics_axis = axes[3]
    u1_steps, _u1_reference = load_scalar("U1 Seed 3", "importance_mean/S")
    ethics_axis.plot(u1_steps, np.zeros_like(u1_steps, dtype=float),
                     color=RUNS["U1 Seed 3"]["color"], linewidth=2.3,
                     label="U1 Seed 3 (not applicable; zero reference)")
    plot_run(ethics_axis, "U2 Seed 2", "importance_mean/E", show_end_value=True)
    setup_axis(ethics_axis, "Relative contribution", (-0.25, 0.05))
    ethics_axis.axhline(0, color="0.35", linewidth=0.8)
    ethics_axis.set_title("Mean Ethics Importance")
    ethics_axis.legend(loc="best")

    for run_name in RUNS:
        row = {"Agent": run_name}
        for tag, label in [
            ("importance_mean/S", "Safety importance"),
            ("importance_mean/T", "Throughput importance"),
            ("importance_mean/F", "Fairness importance"),
        ]:
            _, values = load_scalar(run_name, tag)
            row[f"Final {label}"] = float(values[-1])
            row[f"Mean {label}"] = float(np.mean(values))
        if run_name == "U2 Seed 2":
            _, ethics_values = load_scalar(run_name, "importance_mean/E")
            row["Final Ethics importance"] = float(ethics_values[-1])
            row["Mean Ethics importance"] = float(np.mean(ethics_values))
        else:
            row["Final Ethics importance"] = 0.0
            row["Mean Ethics importance"] = 0.0
        rows.append(row)

    save_figure(figure, "comparison_04_mean_importance")
    save_csv(rows, "comparison_table_04_mean_importance.csv")
    columns = [
        "Agent", "Final Safety importance", "Final Throughput importance",
        "Final Fairness importance", "Final Ethics importance"
    ]
    render_table(
        rows,
        columns,
        ["Agent", "Final S", "Final T", "Final F", "Final E"],
        "Selected-Agent Mean Reward Importance",
        "comparison_table_04_mean_importance",
        formats={column: (lambda value: f"{value:.3f}") for column in columns if column != "Agent"},
    )

# ======================================================
# 5. PPO STABILITY DIRECT COMPARISON
# ======================================================
def create_ppo_comparison():
    settings = [
        ("train/explained_variance", "Explained variance", (-1, 1)),
        ("train/entropy_loss", "Entropy loss", (-1.7, -1.0)),
        ("train/approx_kl", "Approximate KL", (0, 0.03)),
    ]
    figure, axes = plt.subplots(3, 1, figsize=FIG_TRIPLE, sharex=True)
    rows = []

    for axis, (tag, title, ylim) in zip(axes, settings):
        for run_name in RUNS:
            plot_run(axis, run_name, tag, show_end_value=True)
        setup_axis(axis, title, ylim)
        if tag == "train/explained_variance":
            axis.axhline(0, color="0.35", linewidth=0.8)
        axis.set_title(title)
        axis.legend(loc="best")

    for run_name in RUNS:
        row = {"Agent": run_name}
        for tag, label in [
            ("train/explained_variance", "EV"),
            ("train/entropy_loss", "Entropy"),
            ("train/approx_kl", "KL"),
            ("train/clip_fraction", "Clip fraction"),
        ]:
            _, values = load_scalar(run_name, tag)
            row[f"Final {label}"] = float(values[-1])
            row[f"Minimum {label}"] = float(np.min(values))
            row[f"Maximum {label}"] = float(np.max(values))
            row[f"Mean {label}"] = float(np.mean(values))
        rows.append(row)

    save_figure(figure, "comparison_05_ppo_stability")
    save_csv(rows, "comparison_table_05_ppo_stability.csv")
    columns = ["Agent", "Final EV", "Minimum EV", "Maximum EV", "Final Entropy", "Final KL", "Mean KL"]
    render_table(
        rows,
        columns,
        ["Agent", "Final EV", "Min EV", "Max EV", "Final entropy", "Final KL", "Mean KL"],
        "Selected-Agent PPO Stability",
        "comparison_table_05_ppo_stability",
        formats={column: (lambda value: f"{value:.4f}") for column in columns if column != "Agent"},
    )

# ======================================================
# 6. ETHICS PROFILE FOR SELECTED U2 ONLY
# ======================================================
def create_u2_ethics_profile():
    settings = [
        ("ethics_active/E_mean", "Mean ethics score", (-3, 0), None),
        ("ethics_active/violation_lt_-0.5", "Violations below -0.5", (0, 80), lambda y: y * 100),
        ("ethics_active/violation_lt_-1.0", "Severe violations below -1.0", (0, 80), lambda y: y * 100),
    ]
    figure, axes = plt.subplots(3, 1, figsize=FIG_TRIPLE, sharex=True)
    row = {"Agent": "U2 Seed 2"}

    for axis, (tag, title, ylim, transform) in zip(axes, settings):
        plot_run(axis, "U2 Seed 2", tag, transform=transform, show_end_value=True)
        setup_axis(axis, title, ylim)
        if transform:
            axis.yaxis.set_major_locator(MultipleLocator(10))
            axis.yaxis.set_major_formatter(FuncFormatter(pctfmt))
        axis.set_title(f"U2 Seed 2: {title}")
        axis.legend(loc="best")

        _, values = load_scalar("U2 Seed 2", tag)
        if transform:
            values = transform(values)
        row[f"Final {title}"] = float(values[-1])
        row[f"Mean {title}"] = float(np.mean(values))
        row[f"Minimum {title}"] = float(np.min(values))
        row[f"Maximum {title}"] = float(np.max(values))

    save_figure(figure, "comparison_06_u2_ethics_profile")
    save_csv([row], "comparison_table_06_u2_ethics_profile.csv")
    columns = [
        "Agent", "Final Mean ethics score", "Mean Mean ethics score",
        "Mean Violations below -0.5", "Mean Severe violations below -1.0"
    ]
    render_table(
        [row],
        columns,
        ["Agent", "Final E", "Mean E", "Mean < -0.5 (%)", "Mean < -1.0 (%)"],
        "Selected U2 Ethics Profile",
        "comparison_table_06_u2_ethics_profile",
        formats={column: (lambda value: f"{value:.3f}") for column in columns if column != "Agent"},
    )

# ======================================================
# MAIN
# ======================================================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Plot mode: {PLOT_MODE}")
    print(f"Smoothing weight: {SMOOTHING_WEIGHT}")

    create_reward_convergence()
    create_interaction_comparison()
    create_stf_comparison()
    create_importance_comparison()
    create_ppo_comparison()
    create_u2_ethics_profile()

    print("All selected-agent comparison figures and tables created successfully.")


if __name__ == "__main__":
    main()
