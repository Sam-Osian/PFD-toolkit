#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from adjustText import adjust_text
from matplotlib.ticker import PercentFormatter

# ----------------- Tuning knobs -----------------
X_PAD_RIGHT = 10.0
X_MIN_PAD   = 2.0
INIT_DX     = 1
INIT_DY     = 0.00

# Scatter jitter to spread near-duplicates
JITTER_X    = 0.001
JITTER_Y    = 0.0005

# Strength of collision handling
EXP_TEXT    = (1.02, 1.55)
EXP_POINTS  = (1.05, 1.20)
FORCE_TEXT  = 1
FORCE_POINTS= 0.4
N_STEPS     = 1000

# Arrow aesthetics
ARROW = dict(arrowstyle='-', color='lightgray', lw=1, alpha=0.7)

# Colours
BLUE  = "#3B82F6"   # dense
GREEN = "#10B981"   # MoE / sparse
EDGE  = "black"

# Short names for models displayed in labels
SHORT_LABELS = {
    "mistralai/devstral-small": "DevStral Small",
    "mistralai/mistral-large-2411": "Mistral Large",
    "google/gemma-3-4b-it": "Gemma 3 (4B)",
    "deepseek/deepseek-chat-v3-0324": "DeepSeek v3",
    "moonshotai/kimi-k2": "Kimi K2",
    "qwen/qwen3-235b-a22b-2507": "Qwen3 A22B",
    "meta-llama/llama-4-maverick": "Llama 4 Mav.",
    "mistral-nemo:12b": "Mistral Nemo",
    "mistral-small:22b": "Mistral Small (22B)",
    "mistral-small:24b": "Mistral Small (24B)",
    "gemma3:12b": "Gemma 3 (12B)",
    "gemma3:27b": "Gemma 3 (27B)",
    "gemma2:27b": "Gemma 2 (27B)",
    "qwen3:32b": "Qwen3 (32B)",
    "qwen3:30b": "Qwen3 (30B)",
    "qwen2.5:72b": "Qwen 2.5 (72B)",
    "qwen2.5:32b": "Qwen 2.5 (32B)",
    "llava:34b": "LLaVA",
    "phi4:14b": "Phi-4",
    "llama3:70b": "Llama 3",
    "google/gemma-3-12b-it": "Gemma 3 (12B)",
    "cohere/command-a": "Command A",
    "mistralai/codestral-2508": "Codestral"
}

# Baseline references (optional)
FRONTIER_NAMES = {"gpt-4.1": "GPT-4.1"}

# ----------------- MoE identification -----------------
# Explicit MoE models by id in your CSV (exact match)
MOE_EXPLICIT = {
    "deepseek/deepseek-chat-v3-0324",
    "qwen/qwen3-235b-a22b-2507",
    "moonshotai/kimi-k2",
    "meta-llama/llama-4-maverick",
}

# Heuristic keywords that imply MoE/activated-params variants
# e.g., "...-A22B", "...-A3B", or strings that uniquely identify MoE families
import re
MOE_PATTERNS = [
    re.compile(r"(?i)\bA\d+B\b"),         # ...-A22B, -A3B, etc.
    re.compile(r"(?i)\bmaverick\b"),      # Llama 4 Maverick
    re.compile(r"(?i)deepseek.*v3"),      # DeepSeek V3 variants
    re.compile(r"(?i)kimi[-_]?k2"),       # Moonshot K2 variants
]

def is_moe_model(model_id: str) -> bool:
    if model_id in MOE_EXPLICIT:
        return True
    return any(p.search(model_id) for p in MOE_PATTERNS)

# ------------------------------------------------------

def load_data(csv_path: str):
    df = pd.read_csv(csv_path)
    df["local"] = df["local"].astype(str).str.lower()

    local_df = df[df["local"] == "yes"].copy()
    local_df["params"] = pd.to_numeric(local_df["params"], errors="coerce")
    local_df["accuracy"] = pd.to_numeric(local_df["accuracy"], errors="coerce")
    local_df = local_df.dropna(subset=["params", "accuracy"]).reset_index(drop=True)

    # Jitter points slightly to reveal duplicates
    rng = np.random.default_rng(123)
    local_df["params_jit"]   = local_df["params"]   + rng.uniform(-JITTER_X, JITTER_X, size=len(local_df))
    local_df["accuracy_jit"] = local_df["accuracy"] + rng.uniform(-JITTER_Y, JITTER_Y, size=len(local_df))

    # Identify MoE vs dense
    local_df["is_moe"] = local_df["model"].astype(str).apply(is_moe_model)

    # Reference lines (if present)
    ref_df = df[df["model"].isin(FRONTIER_NAMES.keys())][["model", "accuracy"]].copy()
    ref_df["label"] = ref_df["model"].map(FRONTIER_NAMES)
    ref_df.dropna(subset=["accuracy"], inplace=True)

    if local_df.empty:
        raise ValueError("No local models with numeric params+accuracy found.")

    return local_df, ref_df


def make_plot(local_df: pd.DataFrame, ref_df: pd.DataFrame, width=13, height=7, dpi=180):
    fig, ax = plt.subplots(figsize=(width, height), dpi=dpi, constrained_layout=True)

    # Split groups
    moe_df   = local_df[local_df["is_moe"]]
    dense_df = local_df[~local_df["is_moe"]]

    sc_handles = []

    # Dense first (blue)
    sc_dense = None
    if not dense_df.empty:
        sc_dense = ax.scatter(
            dense_df["params_jit"], dense_df["accuracy_jit"],
            s=40, zorder=2, facecolor=BLUE, edgecolor=EDGE, linewidth=0.6, alpha=0.95, label="Dense (all params active)"
        )
        sc_handles.append(sc_dense)

    # MoE second (green)
    sc_moe = None
    if not moe_df.empty:
        sc_moe = ax.scatter(
            moe_df["params_jit"], moe_df["accuracy_jit"],
            s=40, zorder=2, facecolor=GREEN, edgecolor=EDGE, linewidth=0.6, alpha=0.95, label="Sparse MoE (active params)"
        )
        sc_handles.append(sc_moe)

    # Labels & arrows (keep per-point labels)
    rng = np.random.default_rng(42)
    texts = []
    for i, r in local_df.reset_index(drop=True).iterrows():
        label = SHORT_LABELS.get(str(r["model"]), str(r["model"]))
        dx = INIT_DX
        dy = (1 if (i % 2 == 0) else -1) * (INIT_DY * (1.0 + 0.3*rng.random()))
        ann = ax.annotate(
            label,
            xy=(r["params_jit"], r["accuracy_jit"]),
            xytext=(r["params_jit"] + dx, r["accuracy_jit"] + dy),
            textcoords="data",
            ha="left", va="center", fontsize=7, color="black",
            arrowprops=ARROW
        )
        texts.append(ann)

    # Axes, title, grid
    ax.set_xlabel("Active Parameters (B)")
    ax.set_ylabel("Accuracy")
    ax.grid(True, alpha=0.2)
    ax.set_title(
        "Comparing open-source, quantised LLMs vs GPT-4.1 — Accuracy vs Active Parameters (billions)",
        loc="left"
    )

    # Domains with padding
    x_min = float(local_df["params_jit"].min()) - X_MIN_PAD
    x_max = float(local_df["params_jit"].max()) + X_PAD_RIGHT
    y_min = max(0.0, float(local_df["accuracy_jit"].min()) - 0.03)
    y_max = 1.0
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Reference lines
    if not ref_df.empty:
        for _, r in ref_df.iterrows():
            ax.axhline(r["accuracy"], color="grey", lw=0.6, ls=(0,(6,6)), zorder=1, alpha=0.9)
            ax.text(x_max - 0.6, r["accuracy"] + 0.002,
                    f"{r['label']} (Acc {r['accuracy']*100:.1f}%)",
                    ha="right", va="bottom", fontsize=7.5, color="grey")

    # Solve overlaps (include both scatter collections if present)
    add_objs = [h for h in [sc_dense, sc_moe] if h is not None]
    adjust_text(
        texts,
        ax=ax,
        add_objects=add_objs,
        expand_text=EXP_TEXT,
        expand_points=EXP_POINTS,
        force_text=FORCE_TEXT,
        force_points=FORCE_POINTS,
        only_move={"points": "y", "text": "xy"},
        autoalign=True,
        precision=0.01,
        lim=N_STEPS
    )

    # Legend (explicit so colours map to meaning)
    ax.legend(loc="lower right", frameon=True, fontsize=8, title="Architecture")

    # Ticks formatting
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=1))

    # Footnote to reinforce "active params" idea
    fig.text(0.01, 0.01,
             "Note: Green = sparse MoE (subset of experts active per token). X-axis shows active parameters, not total.",
             fontsize=8, color="dimgray")

    return fig, ax


def main():
    p = argparse.ArgumentParser()
    p.add_argument('-i', '--input',  default='model_comparison.csv', help='Path to CSV input')
    p.add_argument('-o', '--output', default='local_models_scatter.png', help='PNG output path')
    p.add_argument('--width',  type=float, default=13.0, help='Figure width (inches)')
    p.add_argument('--height', type=float, default=7.0,  help='Figure height (inches)')
    p.add_argument('--dpi',    type=int,   default=180,  help='Figure DPI')
    args, _unknown = p.parse_known_args()

    local_df, ref_df = load_data(args.input)
    fig, _ = make_plot(local_df, ref_df, width=args.width, height=args.height, dpi=args.dpi)

    out = Path(args.output)
    fig.savefig(out, dpi=args.dpi)
    print(f"Saved chart to {out.resolve()}")


if __name__ == '__main__':
    main()
