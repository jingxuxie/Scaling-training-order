"""Shared publication style; figures are regenerated from the saved results."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cycler import cycler

COLORS = ("#0072B2", "#D55E00", "#009E73", "#AA4499", "#CC9900", "#667788")
EPSILON_COLORS = {
    0: COLORS[0], 1e-6: COLORS[5], 1e-4: COLORS[1],
    1e-3: COLORS[2], 1e-2: COLORS[3], 0.1: COLORS[4],
}
LINESTYLES = ("-", "--", "-.", ":", (0, (5, 1, 1, 1)))


def epsilon_label(value):
    powers = {1e-6: -6, 1e-4: -4, 1e-3: -3}
    number = rf"10^{{{powers[value]}}}" if value in powers else f"{value:g}"
    return rf"$\varepsilon={number}$"


def apply_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.labelpad": 6,
        "axes.edgecolor": "#444444",
        "axes.labelcolor": "#222222",
        "axes.linewidth": 0.75,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "axes.prop_cycle": cycler(color=COLORS),
        "xtick.color": "#444444",
        "ytick.color": "#444444",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.top": False,
        "ytick.right": False,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.minor.size": 2,
        "ytick.minor.size": 2,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "lines.linewidth": 1.8,
        "lines.markersize": 4.5,
        "lines.markeredgewidth": 0.9,
        "legend.frameon": False,
        "legend.fontsize": 9.5,
        "legend.handlelength": 2.4,
        "legend.columnspacing": 1.5,
        "legend.labelspacing": 0.4,
        "legend.borderaxespad": 0,
        "grid.color": "#E1E4E8",
        "grid.linewidth": 0.55,
        "grid.alpha": 0.8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.pad_inches": 0.05,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def legend_above(ax, ncol=2):
    return ax.legend(loc="lower left", bbox_to_anchor=(0, 1.025), ncol=ncol)


def reference_line(ax, value=1, label=None):
    return ax.axhline(value, color="#777777", linestyle=(0, (2, 2)),
                      linewidth=0.9, zorder=1, label=label)


def save_figure(fig, figures, name):
    figures = Path(figures)
    figures.mkdir(parents=True, exist_ok=True)
    for ax in fig.axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(which="both", top=False, right=False)
        ax.grid(axis="y", which="major")
    fig.tight_layout(pad=0.6)
    fig.savefig(figures / f"{name}.pdf", bbox_inches="tight",
                metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(figures / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
