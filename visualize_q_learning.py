
"""
footRein - Offline Q-learning report generator.

Run:
    python visualize_q_learning.py

Reads:
    Scripts/q_table.json

Creates a new timestamped report folder containing:
    all_plots.png
    q_table_heatmap.png
    reward_trend.png
    td_error_and_loss.png
    action_counts.png
    average_q_values.png
    state_exploration.png
    learning_metrics.csv
    q_table_snapshot.json

No live dashboard or continuous refresh.
"""

import csv
import json
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")  # Save figures without opening plot windows.

import matplotlib.pyplot as plt
import numpy as np


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

Q_TABLE_PATH = os.path.join(
    PROJECT_DIR, "Scripts", "q_table.json"
)

OUTPUT_ROOT = os.path.join(
    PROJECT_DIR, "rl_results"
)

ACTIONS = ["nearest", "forward", "open"]
MAX_HEATMAP_STATES = 20
MOVING_AVERAGE_WINDOW = 10


def load_data():
    """Load the latest saved Q-table and training history."""

    with open(Q_TABLE_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data.get("q", {}), dict):
        raise ValueError("The q field in q_table.json must be an object.")

    return data


def moving_average(values, window=MOVING_AVERAGE_WINDOW):
    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        return values

    result = np.empty(len(values), dtype=float)

    for i in range(len(values)):
        start = max(0, i - window + 1)
        result[i] = np.mean(values[start:i + 1])

    return result


def finite_values(values):
    """Convert values to finite floats for plotting."""

    result = []

    for value in values:
        try:
            number = float(value)
            result.append(number if np.isfinite(number) else 0.0)
        except (ValueError, TypeError):
            result.append(0.0)

    return result


def plot_heatmap(ax, q_table):
    """Plot recent states without overlapping text annotations."""

    states = sorted(q_table.keys())[-MAX_HEATMAP_STATES:]

    if not states:
        ax.set_title("1. Q-table heatmap")
        ax.text(0.5, 0.5, "No Q-table states found",
                ha="center", va="center",
                transform=ax.transAxes)
        return None

    matrix = np.array([
        [
            float(q_table[state].get(action, 0.0))
            for action in ACTIONS
        ]
        for state in states
    ], dtype=float)

    # Symmetric scale makes positive and negative values comparable.
    max_abs = max(float(np.max(np.abs(matrix))), 0.01)

    image = ax.imshow(
        matrix,
        aspect="auto",
        cmap="coolwarm",
        vmin=-max_abs,
        vmax=max_abs,
        interpolation="nearest"
    )

    ax.set_xticks(range(len(ACTIONS)))
    ax.set_xticklabels(ACTIONS)
    ax.set_yticks(range(len(states)))
    ax.set_yticklabels(states, fontsize=8)
    ax.set_xlabel("Passing action")
    ax.set_ylabel("State: x-zone, lane, pressure, goal-zone")
    ax.set_title(f"1. Q-table heatmap (latest {len(states)} states)")

    return image


def plot_rewards(ax, history):
    steps = history.get("steps", [])
    rewards = finite_values(history.get("rewards", []))

    count = min(len(steps), len(rewards))

    if count:
        x = steps[:count]
        y = rewards[:count]

        ax.plot(x, y, alpha=0.35, label="Reward per update")
        ax.plot(
            x, moving_average(y),
            label=f"Moving average ({MOVING_AVERAGE_WINDOW})"
        )
        ax.axhline(0, linestyle="--", linewidth=1)
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No reward history available",
                ha="center", va="center",
                transform=ax.transAxes)

    ax.set_title("2. Reward trend")
    ax.set_xlabel("Learning update")
    ax.set_ylabel("Reward")
    ax.grid(True, alpha=0.25)


def plot_td_errors(ax, history):
    steps = history.get("steps", [])
    errors = finite_values(history.get("td_errors", []))
    losses = finite_values(history.get("td_losses", []))

    error_count = min(len(steps), len(errors))
    loss_count = min(len(steps), len(losses))

    if error_count:
        x = steps[:error_count]
        y = errors[:error_count]

        ax.plot(x, y, alpha=0.35, label="TD error")
        ax.plot(x, moving_average(y), label="Smoothed TD error")

    if loss_count:
        ax.plot(
            steps[:loss_count],
            moving_average(losses[:loss_count]),
            label="Smoothed squared TD error"
        )

    if not error_count and not loss_count:
        ax.text(0.5, 0.5, "No TD-error history available",
                ha="center", va="center",
                transform=ax.transAxes)

    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_title("3. TD error and squared TD error")
    ax.set_xlabel("Learning update")
    ax.set_ylabel("Error / squared error")
    ax.legend()
    ax.grid(True, alpha=0.25)


def plot_action_counts(ax, history):
    counts = history.get("action_counts", {})
    values = [int(counts.get(action, 0)) for action in ACTIONS]

    ax.bar(ACTIONS, values)
    ax.set_title("4. Action selection counts")
    ax.set_xlabel("Passing tactic")
    ax.set_ylabel("Number of selections")
    ax.grid(True, axis="y", alpha=0.25)


def plot_mean_q(ax, history):
    steps = history.get("steps", [])
    action_history = history.get("mean_q_by_action", {})

    plotted = False

    for action in ACTIONS:
        values = finite_values(action_history.get(action, []))
        count = min(len(steps), len(values))

        if count:
            ax.plot(
                steps[:count],
                values[:count],
                label=action
            )
            plotted = True

    if not plotted:
        ax.text(0.5, 0.5, "No Q-value history available",
                ha="center", va="center",
                transform=ax.transAxes)

    ax.set_title("5. Average Q-value by action")
    ax.set_xlabel("Learning update")
    ax.set_ylabel("Mean Q-value")
    if plotted:
        ax.legend()
    ax.grid(True, alpha=0.25)


def plot_state_exploration(ax, history):
    steps = history.get("steps", [])
    visited = history.get("visited_states", [])
    count = min(len(steps), len(visited))

    if count:
        ax.plot(steps[:count], visited[:count])
    else:
        ax.text(0.5, 0.5, "No exploration history available",
                ha="center", va="center",
                transform=ax.transAxes)

    ax.set_title("6. State exploration")
    ax.set_xlabel("Learning update")
    ax.set_ylabel("States in Q-table")
    ax.grid(True, alpha=0.25)


def save_metrics_csv(history, output_path):
    """Save aligned learning history as a CSV."""

    fields = [
        "step",
        "reward",
        "td_error",
        "td_squared_error",
        "mean_q",
        "nearest_mean_q",
        "forward_mean_q",
        "open_mean_q",
        "unique_states"
    ]

    columns = {
        "step": history.get("steps", []),
        "reward": history.get("rewards", []),
        "td_error": history.get("td_errors", []),
        "td_squared_error": history.get("td_losses", []),
        "mean_q": history.get("mean_q", []),
        "nearest_mean_q": history.get("mean_q_by_action", {}).get("nearest", []),
        "forward_mean_q": history.get("mean_q_by_action", {}).get("forward", []),
        "open_mean_q": history.get("mean_q_by_action", {}).get("open", []),
        "unique_states": history.get("visited_states", [])
    }

    row_count = max(
        (len(values) for values in columns.values()),
        default=0
    )

    with open(
        output_path, "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        for i in range(row_count):
            writer.writerow({
                name: values[i] if i < len(values) else ""
                for name, values in columns.items()
            })


def generate_report(data, output_dir):
    """Generate and save one static report."""

    q_table = data.get("q", {})
    history = data.get("history", {})
    total_steps = data.get("steps", 0)

    fig, axes = plt.subplots(3, 2, figsize=(16, 11))

    image = plot_heatmap(axes[0, 0], q_table)
    if image is not None:
        fig.colorbar(
            image,
            ax=axes[0, 0],
            fraction=0.046,
            pad=0.04,
            label="Q-value"
        )

    plot_rewards(axes[0, 1], history)
    plot_td_errors(axes[1, 0], history)
    plot_action_counts(axes[1, 1], history)
    plot_mean_q(axes[2, 0], history)
    plot_state_exploration(axes[2, 1], history)

    fig.suptitle(
        f"footRein - Q-learning Report\n"
        f"Updates: {total_steps} | Stored states: {len(q_table)}",
        fontsize=15
    )

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    # Combined image.
    fig.savefig(
        os.path.join(output_dir, "all_plots.png"),
        dpi=160,
        bbox_inches="tight"
    )

    # Individual images.
    plot_names = [
        "q_table_heatmap",
        "reward_trend",
        "td_error_and_loss",
        "action_counts",
        "average_q_values",
        "state_exploration"
    ]

    for ax, name in zip(axes.flat, plot_names):
        fig_one, ax_one = plt.subplots(figsize=(10, 6))

        # Copy the subplot into its own figure.
        # Re-render each plot independently for clean labels.
        if name == "q_table_heatmap":
            image = plot_heatmap(ax_one, q_table)
            if image is not None:
                fig_one.colorbar(image, ax=ax_one, label="Q-value")
        elif name == "reward_trend":
            plot_rewards(ax_one, history)
        elif name == "td_error_and_loss":
            plot_td_errors(ax_one, history)
        elif name == "action_counts":
            plot_action_counts(ax_one, history)
        elif name == "average_q_values":
            plot_mean_q(ax_one, history)
        elif name == "state_exploration":
            plot_state_exploration(ax_one, history)

        fig_one.tight_layout()
        fig_one.savefig(
            os.path.join(output_dir, name + ".png"),
            dpi=160,
            bbox_inches="tight"
        )
        plt.close(fig_one)

    plt.close(fig)

    save_metrics_csv(
        history,
        os.path.join(output_dir, "learning_metrics.csv")
    )

    with open(
        os.path.join(output_dir, "q_table_snapshot.json"),
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(data, file, indent=2)

    # Short text summary for each report.
    with open(
        os.path.join(output_dir, "report_summary.txt"),
        "w",
        encoding="utf-8"
    ) as file:
        file.write(f"Report created: {datetime.now().isoformat()}\n")
        file.write(f"Q-learning updates: {total_steps}\n")
        file.write(f"Stored states: {len(q_table)}\n")
        file.write(
            "Reward history points: "
            f"{len(history.get('rewards', []))}\n"
        )
        file.write(
            "TD-error history points: "
            f"{len(history.get('td_errors', []))}\n"
        )
        file.write(
            "Note: Q-table values and training diagnostics do not "
            "alone prove improved match performance.\n"
        )


def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    timestamp = datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"
    )

    output_dir = os.path.join(
        OUTPUT_ROOT,
        f"report_{timestamp}"
    )

    os.makedirs(output_dir, exist_ok=False)

    if not os.path.exists(Q_TABLE_PATH):
        raise FileNotFoundError(
            f"Q-table not found: {Q_TABLE_PATH}\n"
            "Run the football simulation first."
        )

    data = load_data()

    generate_report(data, output_dir)

    print("\nQ-learning report generated successfully.")
    print("Updates:", data.get("steps", 0))
    print("States:", len(data.get("q", {})))
    print("Output folder:", output_dir)


if __name__ == "__main__":
    main()
