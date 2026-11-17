"""Matplotlib drawings used by the tabs: risk curve, vitals, measurement pattern and contribution gauge."""
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

from backend import core
from backend.config import LABS, VITALS
from gui.style import BACKGROUND, HOSP_A, INK, MUTED, NONSEPTIC, SEPTIC

GREY = "#BBBBBB"
MASK_CMAP = ListedColormap([BACKGROUND, "#3D4F66"])


def risk(ax, curves: dict[str, pd.Series], hour: int, thr: float, onset: float | None) -> None:
    colors = {"risk": SEPTIC, "risk with hidden variables": NONSEPTIC}
    for name, s in curves.items():
        s = s[s.index <= hour]
        ax.plot(s.index, s.values, marker="o", ms=3, color=colors[name], label=name)
    ax.axhline(thr, color=INK, ls="--", lw=1, label=f"threshold {thr:.4f}")
    if onset is not None and onset <= hour + 1:
        ax.axvline(onset, color=SEPTIC, lw=2, label="sepsis onset")
    ax.set(xlim=(1, max(hour, 12)), xlabel="Hour of stay", ylabel="Calibrated risk")
    ax.legend(loc="best", fontsize=7)


def vitals(axes, p: pd.DataFrame, hour: int, greyed: bool) -> None:
    """One small panel per vital: carried-forward step line plus observed dots."""
    d = p[p.hour <= hour]
    right = max(int(p.hour.max()), 12)
    for ax, var in zip(axes, core.PANEL_VITALS):
        ax.step(d.hour, d[var].ffill(), where="post", color=GREY if greyed else MUTED, lw=1.2)
        obs = d.dropna(subset=[var])
        ax.scatter(obs.hour, obs[var], s=10, color=GREY if greyed else SEPTIC, zorder=3)
        ax.set_title(var + (" (not used by M)" if greyed else ""), fontsize=9, color="#999999" if greyed else INK)
        ax.set_xlim(1, right)
        ax.tick_params(labelsize=7)


def mask(ax, p: pd.DataFrame, hour: int) -> None:
    """Which variables were measured at which hour: what model M sees."""
    names = VITALS + [v for v in LABS if p[v].notna().any()]
    d = p[p.hour <= hour]
    m = d[names].notna().to_numpy().T.astype(float)
    ax.imshow(m, aspect="auto", cmap=MASK_CMAP, vmin=0, vmax=1, interpolation="nearest",
              extent=[d.hour.min() - 0.5, d.hour.max() + 0.5, len(names) - 0.5, -0.5])
    step = 2 if len(names) > 20 else 1                  # every second label when many labs were drawn
    ax.set_yticks(range(0, len(names), step), names[::step], fontsize=6)
    ax.set_xlabel("Hour")
    ax.grid(False)


def gauge(ax, shares: list[float]) -> None:
    """One horizontal bar split into values / measurement / statics shares of |SHAP|."""
    left = 0.0
    for share, color, short in zip(shares, [HOSP_A, SEPTIC, GREY], "VMS"):
        ax.barh(0, share, left=left, color=color, height=0.8)
        if share >= 0.15:
            ax.text(left + share / 2, 0, f"{short} {share:.0%}", ha="center", va="center", color="white",
                    fontweight="bold", fontsize=10)
        left += share
    ax.set_xlim(0, 1)
    ax.axis("off")
    ax.set_title("  ".join(f"{n} {v:.0%}" for n, v in zip(["values (V)", "measurement (M)", "statics (S)"], shares)),
                 fontsize=8, fontweight="normal")


def first_alert(s: pd.Series, thr: float) -> int | None:
    hours = s.index[s.values >= thr]
    return int(hours[0]) if len(hours) else None


def max_or_nan(s: pd.Series) -> float:
    return float(s.max()) if len(s) else np.nan
