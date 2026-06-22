#!/usr/bin/env python3
"""Shared helpers for the paper-figure scripts.

visualize_statics.py and visualize_dynamics.py share the material colours, the
log-axis tick formatters, and the PNG+PDF save routine. They keep their own
PAPER_STYLE rcParams (the font sizes are tuned per figure).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import ticker as mticker

# Material colours shared across the statics and dynamics figures.
STEEL_COLOR = "tab:blue"
TPU_COLOR = "tab:orange"


def format_log_n_axis(ax, n_values) -> None:
    """Make a log-scaled N axis show only the explicit N values as ticks."""
    n_list = [int(n) for n in n_values]
    ax.xaxis.set_major_locator(mticker.FixedLocator(n_list))
    ax.xaxis.set_major_formatter(mticker.FixedFormatter([str(n) for n in n_list]))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xlim(min(n_list) * 0.9, max(n_list) * 1.1)


def format_log_y_decimal(ax) -> None:
    """Force a log Y axis to show ~5+ decimal-formatted ticks across the
    visible range (no 10^x labels)."""

    def decimal_fmt(v, _):
        # Trim trailing zeros from a fixed-point representation.
        if v == 0:
            return "0"
        s = f"{v:.10f}".rstrip("0").rstrip(".")
        return s if s else f"{v:g}"

    # Major ticks at every 1, 2, 5 within each decade => ~5-6 ticks per decade.
    ax.yaxis.set_major_locator(
        mticker.LogLocator(base=10.0, subs=(1.0, 2.0, 5.0), numticks=20)
    )
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(decimal_fmt))
    ax.yaxis.set_minor_locator(
        mticker.LogLocator(base=10.0, subs=(3.0, 4.0, 6.0, 7.0, 8.0, 9.0), numticks=40)
    )
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())


def save_figure(fig, prefix: Path, note=None) -> None:
    """Write a figure as both PNG (300 dpi) and PDF, then close it.

    If ``note`` is given, stamp it across the top of the figure (e.g. a dry-run
    marker) so the output cannot be mistaken for a real paper figure.
    """
    if note:
        fig.text(
            0.5,
            0.99,
            note,
            ha="center",
            va="top",
            color="red",
            fontsize=9,
            fontweight="bold",
            zorder=1000,
            bbox=dict(
                boxstyle="round", facecolor="yellow", edgecolor="red", alpha=0.85
            ),
        )
    prefix.parent.mkdir(parents=True, exist_ok=True)
    png = prefix.with_suffix(".png")
    pdf = prefix.with_suffix(".pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")
