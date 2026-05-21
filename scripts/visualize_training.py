"""
Training Visualization
=======================
Generates plots from training history JSON files.

Styles:
    --style dark    Neon-on-dark theme (presentations, demos)
    --style paper   White background, clean serif (academic papers)

Generates:
    1. Loss Curves (Train vs Validation)
    2. Perplexity Curves
    3. Teacher Forcing Schedule
    4. Prediction Evolution (how outputs improve across epochs)
    5. Combined Dashboard (all in one figure)
    6. Cross-Stage Comparison (when using 'all' — shows improvement across stages)

Usage:
    python scripts/visualize_training.py stage1
    python scripts/visualize_training.py all --style paper
"""

import os
import sys
import json
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATA_DIR

# ==========================================
# STYLE DEFINITIONS
# ==========================================
STYLES = {
    "dark": {
        "bg": "#1b1b2f", "card": "#1a1a2e", "grid": "#2a2a3e",
        "text": "#e8e8f0", "dim": "#8888a0",
        "c1": "#6c63ff", "c2": "#ff6b35", "c3": "#00d2ff",
        "c4": "#f472b6", "c5": "#4ade80", "c6": "#fbbf24",
        "font_family": "sans-serif",
        "font_list": ["Nirmala UI", "Mangal", "Arial Unicode MS", "DejaVu Sans"],
        "legend_bg": "#1a1a2e", "legend_edge": "#2a2a3e",
        "table_header_bg": "#6c63ff", "table_header_fg": "white",
        "table_input_bg": "#2a2a4e", "table_input_fg": "#00d2ff",
        "table_cell_bg": "#1a1a2e", "table_cell_fg": "#4ade80",
        "suffix": "",
    },
    "paper": {
        "bg": "white", "card": "white", "grid": "#e0e0e0",
        "text": "#1a1a1a", "dim": "#555555",
        "c1": "#2563eb", "c2": "#dc2626", "c3": "#059669",
        "c4": "#7c3aed", "c5": "#0891b2", "c6": "#d97706",
        "font_family": "serif",
        "font_list": ["Times New Roman", "Nirmala UI", "DejaVu Serif"],
        "legend_bg": "white", "legend_edge": "#cccccc",
        "table_header_bg": "#e5e7eb", "table_header_fg": "#1a1a1a",
        "table_input_bg": "#f9fafb", "table_input_fg": "#1a1a1a",
        "table_cell_bg": "white", "table_cell_fg": "#1a1a1a",
        "suffix": "_paper",
    },
}

S = STYLES["dark"]  # Active style, set in main()


def apply_style(style_name):
    """Apply a named style globally."""
    global S
    S = STYLES[style_name]
    plt.rcParams.update({
        "figure.facecolor": S["bg"],
        "axes.facecolor": S["card"],
        "axes.edgecolor": S["grid"],
        "axes.labelcolor": S["text"],
        "axes.grid": True,
        "grid.color": S["grid"],
        "grid.alpha": 0.5 if style_name == "dark" else 0.7,
        "xtick.color": S["dim"],
        "ytick.color": S["dim"],
        "text.color": S["text"],
        "font.size": 11 if style_name == "dark" else 10,
        "font.family": S["font_family"],
        f"font.{S['font_family']}": S["font_list"],
        "legend.facecolor": S["legend_bg"],
        "legend.edgecolor": S["legend_edge"],
        "legend.fontsize": 10 if style_name == "dark" else 9,
    })


def load_history(stage_name):
    """Load training history JSON for a given stage."""
    path = os.path.join(DATA_DIR, f"{stage_name}_history.json")
    if not os.path.exists(path):
        print(f"ERROR: {path} not found.")
        print(f"Run training first: python scripts/train_{stage_name}.py")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ==========================================
# STAGE LABELS (formal names for paper)
# ==========================================
STAGE_LABELS = {
    "stage1": "Stage 1: Phonetic Foundation",
    "stage2": "Stage 2: Word Variation Fine-Tuning",
    "stage3": "Stage 3: Sentence-Level Context",
}


def stage_title(stage_name):
    return STAGE_LABELS.get(stage_name, stage_name.upper())


# ==========================================
# PLOT FUNCTIONS
# ==========================================
def plot_loss_curves(history, stage_name, ax=None):
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 5))

    epochs = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], color=S["c1"], linewidth=2, label="Train Loss",
            marker="o", markersize=3)
    ax.plot(epochs, history["val_loss"], color=S["c2"], linewidth=2, label="Validation Loss",
            marker="s", markersize=3, linestyle="--" if S == STYLES["paper"] else "-")

    best_epoch = np.argmin(history["val_loss"]) + 1
    best_val = min(history["val_loss"])
    ax.axvline(x=best_epoch, color=S["c5"], linestyle="--", alpha=0.5, linewidth=1)
    ax.annotate(f"Best: {best_val:.3f} (Ep. {best_epoch})",
                xy=(best_epoch, best_val), xytext=(best_epoch + 0.5, best_val + 0.05),
                color=S["c5"], fontsize=8,
                arrowprops=dict(arrowstyle="->", color=S["c5"], lw=1.2))

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title(f"{stage_title(stage_name)} — Loss", fontsize=12, fontweight="bold")
    ax.legend()

    if standalone:
        save_path = os.path.join(DATA_DIR, f"{stage_name}_loss_curves{S['suffix']}.png")
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {save_path}")
        plt.close(fig)


def plot_perplexity(history, stage_name, ax=None):
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 5))

    epochs = range(1, len(history["train_ppl"]) + 1)
    ax.plot(epochs, history["train_ppl"], color=S["c3"], linewidth=2, label="Train PPL",
            marker="o", markersize=3)
    ax.plot(epochs, history["val_ppl"], color=S["c4"], linewidth=2, label="Validation PPL",
            marker="s", markersize=3, linestyle="--" if S == STYLES["paper"] else "-")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Perplexity")
    ax.set_title(f"{stage_title(stage_name)} — Perplexity", fontsize=12, fontweight="bold")
    ax.legend()

    if max(history["train_ppl"]) > 100:
        ax.set_yscale("log")

    if standalone:
        save_path = os.path.join(DATA_DIR, f"{stage_name}_perplexity{S['suffix']}.png")
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {save_path}")
        plt.close(fig)


def plot_teacher_forcing(history, stage_name, ax=None):
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 4))

    epochs = range(1, len(history["tf_ratio"]) + 1)
    ax.fill_between(epochs, history["tf_ratio"], alpha=0.15, color=S["c1"])
    ax.plot(epochs, history["tf_ratio"], color=S["c1"], linewidth=2, marker="o", markersize=4)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Teacher Forcing Ratio")
    ax.set_title(f"{stage_title(stage_name)} — Teacher Forcing Schedule",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(-0.05, 1.05)

    if standalone:
        save_path = os.path.join(DATA_DIR, f"{stage_name}_tf_schedule{S['suffix']}.png")
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {save_path}")
        plt.close(fig)


def plot_prediction_evolution(history, stage_name):
    predictions = history.get("predictions", [])
    if not predictions:
        print("  No prediction snapshots found. Skipping prediction evolution plot.")
        return

    fig, ax = plt.subplots(figsize=(12, max(3, len(predictions[0]["samples"]) * 0.7)))
    ax.axis("off")

    sample_words = [s["input"] for s in predictions[0]["samples"]]
    col_labels = [f"Epoch {p['epoch']}" for p in predictions]
    header = ["Input"] + col_labels

    table_data = []
    for i, word in enumerate(sample_words):
        row = [word]
        for p in predictions:
            row.append(p["samples"][i]["prediction"])
        table_data.append(row)

    table = ax.table(cellText=table_data, colLabels=header, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(S["grid"])
        if row == 0:
            cell.set_facecolor(S["table_header_bg"])
            cell.set_text_props(color=S["table_header_fg"], fontweight="bold")
        elif col == 0:
            cell.set_facecolor(S["table_input_bg"])
            cell.set_text_props(color=S["table_input_fg"], fontweight="bold")
        else:
            cell.set_facecolor(S["table_cell_bg"])
            cell.set_text_props(color=S["table_cell_fg"])

    ax.set_title(f"{stage_title(stage_name)} — Prediction Evolution",
                 fontsize=12, fontweight="bold", pad=20)

    save_path = os.path.join(DATA_DIR, f"{stage_name}_predictions{S['suffix']}.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_dashboard(history, stage_name):
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(f"{stage_title(stage_name)} — Training Dashboard",
                 fontsize=16, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    plot_loss_curves(history, stage_name, ax=ax1)
    plot_perplexity(history, stage_name, ax=ax2)
    plot_teacher_forcing(history, stage_name, ax=ax3)

    # Summary stats
    ax4.axis("off")
    total_epochs = len(history["train_loss"])
    best_epoch = int(np.argmin(history["val_loss"])) + 1
    best_val_loss = min(history["val_loss"])
    best_val_ppl = history["val_ppl"][best_epoch - 1]
    final_train_loss = history["train_loss"][-1]
    overfit_gap = history["val_loss"][-1] - history["train_loss"][-1]

    stats = [
        ("Total Epochs", str(total_epochs)),
        ("Best Epoch", str(best_epoch)),
        ("Best Val Loss", f"{best_val_loss:.4f}"),
        ("Best Val PPL", f"{best_val_ppl:.2f}"),
        ("Final Train Loss", f"{final_train_loss:.4f}"),
        ("Overfit Gap", f"{overfit_gap:+.4f}"),
    ]

    table = ax4.table(
        cellText=stats, colLabels=["Metric", "Value"],
        loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(S["grid"])
        if row == 0:
            cell.set_facecolor(S["table_header_bg"])
            cell.set_text_props(color=S["table_header_fg"], fontweight="bold")
        else:
            cell.set_facecolor(S["table_cell_bg"])
            cell.set_text_props(color=S["table_cell_fg"])

    ax4.set_title("Training Summary", fontsize=12, fontweight="bold", pad=15)

    save_path = os.path.join(DATA_DIR, f"{stage_name}_dashboard{S['suffix']}.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.close(fig)


# ==========================================
# CROSS-STAGE COMPARISON (makes stage2 shine)
# ==========================================
def plot_cross_stage_comparison(histories):
    """Plot all stages on the same axes — shows the full training journey."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    colors = [S["c1"], S["c2"], S["c3"]]
    short_labels = ["Stage 1\n(Phonetics)", "Stage 2\n(Variations)", "Stage 3\n(Sentences)"]

    # --- Panel 1: Final Val Loss per stage (bar chart) ---
    ax = axes[0]
    final_val_losses = []
    for h in histories.values():
        final_val_losses.append(min(h["val_loss"]))

    bars = ax.bar(short_labels, final_val_losses, color=colors, width=0.5, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, final_val_losses):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Best Validation Loss")
    ax.set_title("Best Loss per Stage", fontsize=12, fontweight="bold")

    # --- Panel 2: Final Val PPL per stage ---
    ax = axes[1]
    final_val_ppls = []
    for h in histories.values():
        best_idx = int(np.argmin(h["val_loss"]))
        final_val_ppls.append(h["val_ppl"][best_idx])

    bars = ax.bar(short_labels, final_val_ppls, color=colors, width=0.5, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, final_val_ppls):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Best Validation Perplexity")
    ax.set_title("Best Perplexity per Stage", fontsize=12, fontweight="bold")

    # --- Panel 3: Loss curves stitched across all stages ---
    ax = axes[2]
    offset = 0
    for (name, h), c in zip(histories.items(), colors):
        epochs = range(offset + 1, offset + len(h["val_loss"]) + 1)
        label = name.replace("stage", "Stage ")
        ax.plot(epochs, h["val_loss"], color=c, linewidth=2, label=f"{label} (val)",
                marker="s", markersize=2)
        ax.plot(epochs, h["train_loss"], color=c, linewidth=1.2, alpha=0.5,
                linestyle="--", label=f"{label} (train)")
        # Draw stage boundary
        if offset > 0:
            ax.axvline(x=offset + 0.5, color=S["dim"], linestyle=":", linewidth=1)
        offset += len(h["val_loss"])

    ax.set_xlabel("Cumulative Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Journey Across All Stages", fontsize=12, fontweight="bold")
    ax.legend(fontsize=7, ncol=2)

    fig.suptitle("Cross-Stage Training Comparison", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()

    save_path = os.path.join(DATA_DIR, f"cross_stage_comparison{S['suffix']}.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\n  Saved: {save_path}")
    plt.close(fig)


# ==========================================
# ORCHESTRATION
# ==========================================
def visualize_stage(stage_name):
    print(f"\n{'='*50}")
    print(f"  Generating {stage_title(stage_name)} Visualizations")
    print(f"{'='*50}\n")

    history = load_history(stage_name)
    if history is None:
        return None

    print(f"  Loaded {len(history['train_loss'])} epochs of training data.\n")

    print("  [1/5] Loss Curves...")
    plot_loss_curves(history, stage_name)

    print("  [2/5] Perplexity...")
    plot_perplexity(history, stage_name)

    print("  [3/5] Teacher Forcing Schedule...")
    plot_teacher_forcing(history, stage_name)

    print("  [4/5] Prediction Evolution...")
    plot_prediction_evolution(history, stage_name)

    print("  [5/5] Combined Dashboard...")
    plot_dashboard(history, stage_name)

    print(f"\n✓ All {stage_name} visualizations saved to: {DATA_DIR}")
    return history


def main():
    parser = argparse.ArgumentParser(description="Visualize training history")
    parser.add_argument("stage", nargs="?", default="stage1",
                        choices=["stage1", "stage2", "stage3", "all"],
                        help="Which stage to visualize (default: stage1)")
    parser.add_argument("--style", default="dark", choices=["dark", "paper"],
                        help="Plot style: 'dark' for presentations, 'paper' for publications")
    args = parser.parse_args()

    apply_style(args.style)
    print(f"\n  Style: {args.style.upper()}")

    if args.stage == "all":
        histories = {}
        for stage in ["stage1", "stage2", "stage3"]:
            h = visualize_stage(stage)
            if h is not None:
                histories[stage] = h

        if len(histories) >= 2:
            print(f"\n{'='*50}")
            print(f"  Generating Cross-Stage Comparison")
            print(f"{'='*50}")
            plot_cross_stage_comparison(histories)
    else:
        visualize_stage(args.stage)


if __name__ == "__main__":
    main()
