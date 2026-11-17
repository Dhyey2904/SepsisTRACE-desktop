"""Colours and the matplotlib style used by every chart in the app."""
import matplotlib

SEPTIC = "#C4705E"
NONSEPTIC = "#8FA7C4"
BACKGROUND = "#FAF8F4"
INK = "#2B2B2B"
MUTED = "#6B6B6B"
HOSP_A = "#3D5A80"
GOOD, BAD = "#2E7D32", "#C62828"

matplotlib.rcParams.update({
    "figure.facecolor": BACKGROUND,
    "axes.facecolor": BACKGROUND,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "lines.linewidth": 2,
    "font.size": 9,
})
