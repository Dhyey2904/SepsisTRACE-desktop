"""Shared Tkinter pieces: chart canvas, data table, metric tile, multi-select list, error popup, Tab base class."""
import functools
import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from backend import core
from gui.style import BAD, BACKGROUND, GOOD, INK, MUTED


def fmt(x, spec: str = ".3f") -> str:
    return "—" if x is None or np.isnan(x) else format(x, spec)


def safe(func):
    """Show any error as a popup instead of a traceback in the terminal."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as exc:          # noqa: BLE001 - every error goes to the user
            messagebox.showerror("Error", f"{type(exc).__name__}: {exc}")
    return wrapper


# ------------------------------------------------------------- cached model calls
@functools.lru_cache(maxsize=64)
def hidden_curve(pid: str, hide: tuple, model: str, fset: str, horizon: int) -> pd.Series:
    """Live risk curve with the listed variables treated as never measured."""
    return core.predict_with_hidden(core.patient(pid), list(hide), model, fset, horizon)


@functools.lru_cache(maxsize=256)
def console(model: str, fset: str, horizon: int, threshold: float) -> dict:
    return core.threshold_console(model, fset, horizon, threshold)


# ------------------------------------------------------------------------ widgets
class Chart(ttk.Frame):
    """A matplotlib figure inside a Tkinter frame. Resizes with the window."""

    def __init__(self, parent, width: float = 6, height: float = 3):
        super().__init__(parent)
        self.fig = Figure(figsize=(width, height), dpi=90, facecolor=BACKGROUND)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def axes(self, nrows: int = 1, ncols: int = 1, **kw) -> np.ndarray:
        self.fig.clear()
        return self.fig.subplots(nrows, ncols, squeeze=False, **kw)

    def draw(self) -> None:
        self.fig.tight_layout()
        self.canvas.draw_idle()


class Table(ttk.Frame):
    """Treeview that shows a DataFrame."""

    def __init__(self, parent, height: int = 6):
        super().__init__(parent)
        self.tree = ttk.Treeview(self, show="headings", height=height)
        bar = ttk.Scrollbar(self, command=self.tree.yview)
        self.tree.configure(yscrollcommand=bar.set)
        self.tree.pack(side="left", fill="both", expand=True)
        bar.pack(side="right", fill="y")

    def show(self, df: pd.DataFrame, index: bool = False, digits: int = 3) -> None:
        if index:
            df = df.reset_index()
        cols = [str(c) for c in df.columns]
        self.tree.delete(*self.tree.get_children())
        self.tree.configure(columns=cols)
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=max(70, 9 * len(c)), anchor="center")
        for row in df.itertuples(index=False):
            self.tree.insert("", "end", values=[self._cell(v, digits) for v in row])

    @staticmethod
    def _cell(v, digits: int) -> str:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "—"
        return f"{v:.{digits}f}" if isinstance(v, (float, np.floating)) else str(v)


class Metric(ttk.Frame):
    """Label, big value and an optional coloured change, like a dashboard tile."""

    def __init__(self, parent, label: str):
        super().__init__(parent, padding=(8, 4))
        self.label = ttk.Label(self, text=label, foreground=MUTED)
        self.value = ttk.Label(self, text="—", font=("Helvetica", 20, "bold"), foreground=INK)
        self.delta = ttk.Label(self, text="", font=("Helvetica", 10))
        for w in (self.label, self.value, self.delta):
            w.pack(anchor="w")

    def set(self, value: str, delta: float | None = None, delta_text: str | None = None, label: str | None = None) -> None:
        self.value.configure(text=value)
        if label:
            self.label.configure(text=label)
        if delta is None or np.isnan(delta):
            self.delta.configure(text="")
        else:
            arrow = "▲" if delta > 0 else "▼" if delta < 0 else "■"
            self.delta.configure(text=f"{arrow} {delta_text or f'{delta:+.3f}'}",
                                 foreground=GOOD if delta > 0 else BAD if delta < 0 else MUTED)


class MultiSelect(ttk.Frame):
    """Listbox where several items can be ticked; calls on_change with the chosen items."""

    def __init__(self, parent, items: list[str], on_change, height: int = 5):
        super().__init__(parent)
        self.items = items
        self.box = tk.Listbox(self, selectmode="multiple", height=height, exportselection=False)
        for it in items:
            self.box.insert("end", it)
        bar = ttk.Scrollbar(self, command=self.box.yview)
        self.box.configure(yscrollcommand=bar.set)
        self.box.pack(side="left", fill="both", expand=True)
        bar.pack(side="left", fill="y")
        self.box.bind("<<ListboxSelect>>", lambda e: on_change(self.selected()))

    def selected(self) -> list[str]:
        return [self.items[i] for i in self.box.curselection()]

    def clear(self) -> None:
        self.box.selection_clear(0, "end")
        self.box.event_generate("<<ListboxSelect>>")


class Tab(ttk.Frame):
    """Base class for the notebook tabs. `app` holds the sidebar settings."""

    def __init__(self, parent, app):
        super().__init__(parent, padding=10)
        self.app = app

    def refresh(self) -> None:
        """Redraw for the current sidebar settings."""
