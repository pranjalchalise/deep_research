#!/usr/bin/env python3
"""
Generate architecture diagrams for both pipelines.

Usage:
    python docs/generate_diagrams.py

Outputs:
    docs/standard_pipeline.png
    docs/advanced_pipeline.png
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np


# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------
COLORS = {
    "start_end": "#1a1a2e",
    "shared": "#3b82f6",       # blue - shared nodes
    "single": "#8b5cf6",       # purple - single-agent nodes
    "multi": "#10b981",        # green - multi-agent nodes
    "trust": "#f59e0b",        # amber - trust engine nodes
    "discovery": "#6366f1",    # indigo - discovery nodes
    "iterative": "#ef4444",    # red - iterative/loop nodes
    "text_light": "#ffffff",
    "text_dark": "#1a1a2e",
    "bg": "#ffffff",
    "arrow": "#64748b",
    "loop_arrow": "#ef4444",
    "label_bg": "#f8fafc",
}

FONT = "sans-serif"


def draw_box(ax, x, y, text, color, width=1.6, height=0.5, fontsize=9,
             text_color="white", alpha=1.0, style="round,pad=0.1"):
    """Draw a rounded rectangle with centered text."""
    box = FancyBboxPatch(
        (x - width / 2, y - height / 2), width, height,
        boxstyle=style,
        facecolor=color, edgecolor="none", alpha=alpha,
        zorder=2,
    )
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color=text_color, fontfamily=FONT, zorder=3)
    return box


def draw_diamond(ax, x, y, text, color, size=0.45, fontsize=7):
    """Draw a diamond (decision) shape."""
    d = size
    diamond = plt.Polygon(
        [(x, y + d), (x + d, y), (x, y - d), (x - d, y)],
        facecolor=color, edgecolor="none", alpha=0.85, zorder=2,
    )
    ax.add_patch(diamond)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color="white", fontfamily=FONT, zorder=3)


def draw_arrow(ax, x1, y1, x2, y2, color=None, style="-|>", lw=1.5,
               connectionstyle="arc3,rad=0", label="", label_fontsize=7,
               label_offset=(0.05, 0.05)):
    """Draw an arrow between two points."""
    if color is None:
        color = COLORS["arrow"]
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, color=color, lw=lw,
        connectionstyle=connectionstyle,
        mutation_scale=12, zorder=1,
    )
    ax.add_patch(arrow)
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx + label_offset[0], my + label_offset[1], label,
                fontsize=label_fontsize,
                color=color, fontfamily=FONT, fontstyle="italic",
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.15", facecolor=COLORS["label_bg"],
                          edgecolor="none", alpha=0.9))


def draw_circle(ax, x, y, text, color, radius=0.25, fontsize=9):
    """Draw a circle (start/end)."""
    circle = plt.Circle((x, y), radius, facecolor=color, edgecolor="none",
                         zorder=2)
    ax.add_patch(circle)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color="white", fontfamily=FONT, zorder=3)


# ============================================================================
# STANDARD PIPELINE DIAGRAM
# ============================================================================

def generate_standard_pipeline():
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(-1.5, 13.5)
    ax.set_ylim(-1.5, 9)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor(COLORS["bg"])

    # Title
    ax.text(6, 8.5, "Standard Pipeline", fontsize=22, fontweight="bold",
            ha="center", va="center", fontfamily=FONT, color=COLORS["text_dark"])
    ax.text(6, 8.0, "src/pipeline/  |  Single-agent & multi-agent modes",
            fontsize=11, ha="center", va="center", fontfamily=FONT,
            color="#64748b")

    # --- Shared top row ---
    draw_circle(ax, 0.5, 7.0, "START", COLORS["start_end"])
    draw_box(ax, 2.5, 7.0, "understand", COLORS["shared"])
    draw_diamond(ax, 4.8, 7.0, "ambig?", COLORS["shared"], size=0.4)

    # clarify
    draw_box(ax, 4.8, 5.7, "clarify", COLORS["shared"], width=1.4, height=0.45, fontsize=9)
    ax.text(4.8, 5.3, "interrupt (HITL)", fontsize=7, ha="center",
            color="#64748b", fontstyle="italic", fontfamily=FONT)

    # plan
    draw_box(ax, 7.2, 7.0, "plan", COLORS["shared"])
    draw_diamond(ax, 9.2, 7.0, "mode?", COLORS["shared"], size=0.4)

    # Top row arrows
    draw_arrow(ax, 0.75, 7.0, 1.7, 7.0)
    draw_arrow(ax, 3.3, 7.0, 4.35, 7.0)
    draw_arrow(ax, 5.25, 7.0, 6.4, 7.0, label="clear")
    draw_arrow(ax, 4.8, 6.55, 4.8, 5.95, label="yes", label_offset=(-0.4, 0))
    # clarify -> plan
    draw_arrow(ax, 5.5, 5.7, 6.4, 7.0, connectionstyle="arc3,rad=-0.3")
    draw_arrow(ax, 8.0, 7.0, 8.75, 7.0)

    # =========================================================
    # SINGLE-AGENT PATH (left)
    # =========================================================
    sa_x = 3.0

    draw_box(ax, sa_x, 4.5, "search_and_extract", COLORS["single"], width=2.2)
    draw_box(ax, sa_x, 3.2, "detect_gaps", COLORS["single"], width=1.9)

    # Mode -> single
    draw_arrow(ax, 9.2, 6.55, sa_x + 0.5, 4.75, label="single",
               connectionstyle="arc3,rad=0.4", label_offset=(-0.8, 0.2))

    draw_arrow(ax, sa_x, 4.25, sa_x, 3.45)

    # Loop: gaps -> search_and_extract
    draw_arrow(ax, 2.05, 3.2, 1.2, 3.2, color=COLORS["loop_arrow"],
               connectionstyle="arc3,rad=0")
    draw_arrow(ax, 1.2, 3.2, 1.2, 4.5, color=COLORS["loop_arrow"],
               connectionstyle="arc3,rad=0")
    draw_arrow(ax, 1.2, 4.5, 1.9, 4.5, color=COLORS["loop_arrow"])
    ax.text(0.55, 3.85, "gaps\nfound", fontsize=7, ha="center",
            color=COLORS["loop_arrow"], fontfamily=FONT, fontstyle="italic")

    # =========================================================
    # MULTI-AGENT PATH (right)
    # =========================================================
    ma_x = 9.5

    draw_box(ax, ma_x, 5.3, "orchestrate", COLORS["multi"], width=1.8)
    ax.text(ma_x, 4.9, "Send()", fontsize=7, ha="center",
            color="#064e3b", fontfamily=FONT, fontstyle="italic",
            fontweight="bold")

    # Mode -> multi
    draw_arrow(ax, 9.65, 7.0, 9.65, 5.55, label="multi",
               connectionstyle="arc3,rad=0.15", label_offset=(0.45, 0))

    # Workers
    for i, offset in enumerate([-1.6, 0, 1.6]):
        draw_box(ax, ma_x + offset, 4.0, f"worker {i+1}", COLORS["multi"],
                 width=1.3, height=0.4, fontsize=8)
        draw_arrow(ax, ma_x + offset * 0.6, 4.65, ma_x + offset, 4.22,
                   color=COLORS["multi"])

    draw_box(ax, ma_x, 2.8, "collect", COLORS["multi"], width=1.5)
    draw_box(ax, ma_x, 1.7, "synthesize", COLORS["multi"], width=1.6)

    for offset in [-1.6, 0, 1.6]:
        draw_arrow(ax, ma_x + offset, 3.78, ma_x, 3.05, color=COLORS["multi"])

    draw_arrow(ax, ma_x, 2.55, ma_x, 1.95)

    # Loop: synthesize -> orchestrate
    draw_arrow(ax, 11.5, 1.7, 12.2, 1.7, color=COLORS["loop_arrow"],
               connectionstyle="arc3,rad=0")
    draw_arrow(ax, 12.2, 1.7, 12.2, 5.3, color=COLORS["loop_arrow"],
               connectionstyle="arc3,rad=0")
    draw_arrow(ax, 12.2, 5.3, 10.4, 5.3, color=COLORS["loop_arrow"])
    ax.text(12.7, 3.5, "needs more\nresearch", fontsize=7, ha="center",
            color=COLORS["loop_arrow"], fontfamily=FONT, fontstyle="italic")

    # =========================================================
    # SHARED BOTTOM (verify -> write -> END)
    # =========================================================
    draw_box(ax, 6.2, 1.2, "verify", COLORS["shared"], width=1.5)
    draw_box(ax, 6.2, 0.0, "write_report", COLORS["shared"], width=1.8)
    draw_circle(ax, 6.2, -1.0, "END", COLORS["start_end"])

    # detect_gaps -> verify
    draw_arrow(ax, sa_x + 0.6, 2.95, 5.45, 1.45, label="ready",
               connectionstyle="arc3,rad=-0.15")

    # synthesize -> verify
    draw_arrow(ax, ma_x - 0.5, 1.45, 6.95, 1.2,
               connectionstyle="arc3,rad=0.15", label="done")

    draw_arrow(ax, 6.2, 0.95, 6.2, 0.25)
    draw_arrow(ax, 6.2, -0.25, 6.2, -0.75)

    # =========================================================
    # Legend
    # =========================================================
    legend_y = -1.2
    legend_items = [
        ("Shared nodes", COLORS["shared"]),
        ("Single-agent path", COLORS["single"]),
        ("Multi-agent path", COLORS["multi"]),
        ("Loop / retry", COLORS["loop_arrow"]),
    ]
    for i, (label, color) in enumerate(legend_items):
        lx = 0.5 + i * 3.0
        draw_box(ax, lx, legend_y, "", color, width=0.35, height=0.2)
        ax.text(lx + 0.3, legend_y, label, fontsize=8, ha="left",
                va="center", fontfamily=FONT, color=COLORS["text_dark"])

    plt.tight_layout()
    plt.savefig("docs/standard_pipeline.png", dpi=200, bbox_inches="tight",
                facecolor=COLORS["bg"], edgecolor="none")
    plt.close()
    print("Saved docs/standard_pipeline.png")


# ============================================================================
# ADVANCED PIPELINE DIAGRAM
# ============================================================================

def generate_advanced_pipeline():
    fig, ax = plt.subplots(1, 1, figsize=(16, 14))
    ax.set_xlim(-1.5, 15.5)
    ax.set_ylim(-3.5, 12)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor(COLORS["bg"])

    # Title
    ax.text(7, 11.5, "Advanced Pipeline", fontsize=22, fontweight="bold",
            ha="center", va="center", fontfamily=FONT, color=COLORS["text_dark"])
    ax.text(7, 11.0, "src/advanced/  |  Trust engine + multi-agent orchestration",
            fontsize=11, ha="center", va="center", fontfamily=FONT,
            color="#64748b")

    # =========================================================
    # DISCOVERY PHASE
    # =========================================================
    phase_label_style = dict(fontsize=9, fontfamily=FONT, color="#94a3b8",
                             fontweight="bold", ha="left", va="center")

    ax.text(-0.8, 10.2, "DISCOVERY", **phase_label_style)

    draw_circle(ax, 0.5, 9.8, "START", COLORS["start_end"])
    draw_box(ax, 2.5, 9.8, "analyzer", COLORS["discovery"])
    draw_box(ax, 5.0, 9.8, "discovery", COLORS["discovery"])
    draw_diamond(ax, 7.5, 9.8, "conf?", COLORS["discovery"], size=0.4)

    draw_box(ax, 6.0, 8.5, "clarify", COLORS["discovery"], width=1.5, height=0.45, fontsize=9)
    ax.text(6.0, 8.1, "interrupt (HITL)", fontsize=7, ha="center",
            color="#64748b", fontstyle="italic", fontfamily=FONT)
    draw_box(ax, 9.5, 8.5, "auto_refine", COLORS["discovery"], width=1.6, height=0.45, fontsize=9)

    # Discovery arrows
    draw_arrow(ax, 0.75, 9.8, 1.7, 9.8)
    draw_arrow(ax, 3.3, 9.8, 4.2, 9.8)
    draw_arrow(ax, 5.8, 9.8, 7.05, 9.8)
    draw_arrow(ax, 7.5, 9.35, 6.0, 8.75, label="low",
               label_offset=(-0.45, 0))
    draw_arrow(ax, 7.95, 9.8, 9.5, 9.8)
    ax.text(8.7, 10.0, "high", fontsize=7, ha="center", color="#64748b",
            fontstyle="italic", fontfamily=FONT)
    draw_arrow(ax, 7.95, 9.55, 9.5, 8.75, label="medium",
               connectionstyle="arc3,rad=-0.15", label_offset=(0.3, 0.1))

    # =========================================================
    # PLANNING
    # =========================================================
    ax.text(-0.8, 7.5, "PLANNING", **phase_label_style)

    draw_box(ax, 4.0, 7.3, "planner", COLORS["shared"], width=1.6)

    # Connections into planner
    draw_arrow(ax, 9.5, 9.55, 4.8, 7.55, connectionstyle="arc3,rad=0.4",
               label="high conf", label_offset=(1.5, 0.5))
    draw_arrow(ax, 6.0, 8.25, 4.8, 7.55, connectionstyle="arc3,rad=-0.15")
    draw_arrow(ax, 9.5, 8.25, 4.8, 7.55, connectionstyle="arc3,rad=0.15")

    # =========================================================
    # RESEARCH PHASE
    # =========================================================
    ax.text(-0.8, 6.3, "RESEARCH", **phase_label_style)

    draw_box(ax, 4.0, 6.0, "orchestrator", COLORS["multi"], width=1.8)
    ax.text(4.0, 5.6, "Send()", fontsize=7, ha="center",
            color="#064e3b", fontfamily=FONT, fontstyle="italic",
            fontweight="bold")

    draw_arrow(ax, 4.0, 7.05, 4.0, 6.25)

    # Subagents
    for i, offset in enumerate([-2.2, 0, 2.2]):
        draw_box(ax, 4.0 + offset, 4.7, f"subagent {i+1}", COLORS["multi"],
                 width=1.5, height=0.45, fontsize=8)
        draw_arrow(ax, 4.0 + offset * 0.55, 5.5, 4.0 + offset, 4.95,
                   color=COLORS["multi"])

    draw_box(ax, 4.0, 3.5, "synthesizer", COLORS["multi"], width=1.7)

    for offset in [-2.2, 0, 2.2]:
        draw_arrow(ax, 4.0 + offset, 4.45, 4.0, 3.75, color=COLORS["multi"])

    # =========================================================
    # ITERATIVE LOOP
    # =========================================================
    ax.text(-0.8, 2.6, "ITERATION", **phase_label_style)

    draw_box(ax, 4.0, 2.3, "gap_detector", COLORS["iterative"], width=1.8)
    draw_box(ax, 0.8, 2.3, "backtrack", COLORS["iterative"], width=1.5, height=0.45, fontsize=8)

    draw_arrow(ax, 4.0, 3.25, 4.0, 2.55)

    # gap_detector -> backtrack
    draw_arrow(ax, 3.1, 2.3, 1.55, 2.3, color=COLORS["loop_arrow"],
               label="dead end", label_offset=(0, 0.25))

    # backtrack -> orchestrator (loop left side)
    draw_arrow(ax, 0.05, 2.3, -0.6, 2.3, color=COLORS["loop_arrow"])
    draw_arrow(ax, -0.6, 2.3, -0.6, 6.0, color=COLORS["loop_arrow"],
               connectionstyle="arc3,rad=0")
    draw_arrow(ax, -0.6, 6.0, 3.1, 6.0, color=COLORS["loop_arrow"])
    ax.text(-1.1, 4.2, "retry with\nnew angle", fontsize=7, ha="center",
            color=COLORS["loop_arrow"], fontfamily=FONT, fontstyle="italic")

    # gap_detector -> orchestrator (loop right side)
    draw_arrow(ax, 4.9, 2.3, 7.2, 2.3, color=COLORS["loop_arrow"])
    draw_arrow(ax, 7.2, 2.3, 7.2, 6.0, color=COLORS["loop_arrow"],
               connectionstyle="arc3,rad=0")
    draw_arrow(ax, 7.2, 6.0, 4.9, 6.0, color=COLORS["loop_arrow"])
    ax.text(7.8, 4.2, "gaps\nfound", fontsize=7, ha="center",
            color=COLORS["loop_arrow"], fontfamily=FONT, fontstyle="italic")

    # =========================================================
    # REDUCE
    # =========================================================
    draw_box(ax, 4.0, 1.0, "reduce", COLORS["shared"], width=1.5)
    draw_arrow(ax, 4.0, 2.05, 4.0, 1.25, label="ready")

    # =========================================================
    # TRUST ENGINE (right column)
    # =========================================================
    ax.text(9.5, 6.3, "TRUST ENGINE", **phase_label_style)

    te_x = 11.5
    trust_nodes = [
        (te_x, 5.6, "credibility"),
        (te_x, 4.5, "ranker"),
        (te_x, 3.4, "claims"),
        (te_x, 2.3, "cite"),
        (te_x, 1.2, "span_verify"),
        (te_x, 0.1, "cross_validate"),
        (te_x, -1.0, "confidence_score"),
    ]

    for x, y, name in trust_nodes:
        draw_box(ax, x, y, name, COLORS["trust"], width=2.0, fontsize=8.5)

    for i in range(len(trust_nodes) - 1):
        draw_arrow(ax, trust_nodes[i][0], trust_nodes[i][1] - 0.25,
                   trust_nodes[i+1][0], trust_nodes[i+1][1] + 0.25,
                   color=COLORS["trust"])

    # reduce -> credibility (bridge arrow)
    draw_arrow(ax, 4.75, 1.0, 10.5, 5.6,
               connectionstyle="arc3,rad=-0.35", lw=2.0,
               color=COLORS["trust"])

    # Bracket annotation
    bracket_x = te_x + 1.4
    ax.plot([bracket_x, bracket_x], [-1.0, 5.6], color=COLORS["trust"],
            linewidth=1.5, alpha=0.4, zorder=0)
    ax.plot([bracket_x - 0.1, bracket_x], [5.6, 5.6], color=COLORS["trust"],
            linewidth=1.5, alpha=0.4, zorder=0)
    ax.plot([bracket_x - 0.1, bracket_x], [-1.0, -1.0], color=COLORS["trust"],
            linewidth=1.5, alpha=0.4, zorder=0)
    ax.text(bracket_x + 0.2, 2.3, "7 nodes\n(or 2 in\nbatched\nmode)",
            fontsize=7.5, ha="left", va="center", color=COLORS["trust"],
            fontfamily=FONT, fontstyle="italic")

    # =========================================================
    # WRITE + END
    # =========================================================
    draw_box(ax, te_x, -2.2, "write", COLORS["shared"], width=1.5)
    draw_circle(ax, te_x, -3.2, "END", COLORS["start_end"])

    draw_arrow(ax, te_x, -1.25, te_x, -1.95, color=COLORS["trust"])
    draw_arrow(ax, te_x, -2.45, te_x, -2.95)

    # =========================================================
    # Phase separator lines
    # =========================================================
    for y_line in [9.2, 6.8, 1.6]:
        ax.axhline(y=y_line, color="#e2e8f0", linewidth=0.8, linestyle="--",
                   xmin=0.02, xmax=0.98, zorder=0)

    # =========================================================
    # Legend
    # =========================================================
    legend_y = -3.2
    legend_items = [
        ("Discovery", COLORS["discovery"]),
        ("Shared", COLORS["shared"]),
        ("Multi-agent", COLORS["multi"]),
        ("Iterative / loop", COLORS["iterative"]),
        ("Trust engine", COLORS["trust"]),
    ]
    for i, (label, color) in enumerate(legend_items):
        lx = 0.5 + i * 2.7
        draw_box(ax, lx, legend_y, "", color, width=0.35, height=0.2)
        ax.text(lx + 0.3, legend_y, label, fontsize=8, ha="left",
                va="center", fontfamily=FONT, color=COLORS["text_dark"])

    plt.tight_layout()
    plt.savefig("docs/advanced_pipeline.png", dpi=200, bbox_inches="tight",
                facecolor=COLORS["bg"], edgecolor="none")
    plt.close()
    print("Saved docs/advanced_pipeline.png")


if __name__ == "__main__":
    generate_standard_pipeline()
    generate_advanced_pipeline()
