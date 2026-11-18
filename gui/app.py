"""Main window: sidebar settings (patient, model, feature set, horizon, threshold) and the three tabs."""
import tkinter as tk
from tkinter import ttk

import numpy as np

from backend import core
from gui.style import BACKGROUND, MUTED
from gui.tabs import ChallengeTab, ConsoleTab, ExplorerTab

FEATURE_LABELS = {"V": "V values", "M": "M measurement", "VM": "V+M"}
TABS = [("Patient explorer", ExplorerTab), ("Threshold console", ConsoleTab), ("Clinician challenge", ChallengeTab)]


class SepsisApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SepsisTRACE")
        self.geometry("1440x900")
        self.minsize(1150, 760)
        self.configure(background=BACKGROUND)
        style = ttk.Style(self)
        style.theme_use("clam")
        for name in ("TFrame", "TLabel", "TRadiobutton", "TCheckbutton", "TLabelframe", "TLabelframe.Label"):
            style.configure(name, background=BACKGROUND)
        style.configure("TNotebook", background=BACKGROUND)
        self._pending = None
        missing = core.missing_files()
        if missing:
            self.show_missing(missing)
            return
        self.patients = core.load_demo().groupby("patient_id").category.first().sort_values()
        self.thresholds: dict[tuple, float] = {}            # threshold chosen for each model/set/horizon
        self.build_sidebar()
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(side="left", fill="both", expand=True, padx=(0, 8), pady=8)
        self.tabs = []
        for title, cls in TABS:
            tab = cls(self.notebook, self)
            self.notebook.add(tab, text=title)
            self.tabs.append(tab)
        self.notebook.bind("<<NotebookTabChanged>>", lambda e: self.schedule(self.current().refresh))
        self.on_settings()

    # ------------------------------------------------------------------ sidebar
    def build_sidebar(self) -> None:
        side = ttk.Frame(self, padding=12, width=280)
        side.pack(side="left", fill="y")
        side.pack_propagate(False)
        ttk.Label(side, text="SepsisTRACE", font=("Helvetica", 20, "bold")).pack(anchor="w")
        ttk.Label(side, text="Does the pattern of measurement predict sepsis? Test-split patients only.",
                  foreground=MUTED, wraplength=250).pack(anchor="w", pady=(0, 8))

        ttk.Label(side, text="Patient", font=("Helvetica", 11, "bold")).pack(anchor="w", pady=(8, 0))
        self.labels = [f"{pid} — {cat}" for pid, cat in self.patients.items()]
        self.patient_var = tk.StringVar(value=self.labels[0])
        box = ttk.Combobox(side, textvariable=self.patient_var, values=self.labels, state="readonly", width=34)
        box.pack(anchor="w", fill="x")
        box.bind("<<ComboboxSelected>>", lambda e: self.on_settings())

        self.model_var = tk.StringVar(value="xgb")
        self.fset_var = tk.StringVar(value="VM")
        self.horizon_var = tk.IntVar(value=6)
        for title, var, options in (("Model (2)", self.model_var, [(m, m) for m in core.MODELS]),
                                    ("Feature set (3)", self.fset_var, [(FEATURE_LABELS[s], s) for s in core.FEATURE_SETS]),
                                    ("Horizon (hours ahead)", self.horizon_var, [(f"{h}h", h) for h in core.HORIZONS])):
            frame = ttk.LabelFrame(side, text=title, padding=4)
            frame.pack(anchor="w", fill="x", pady=(10, 0))
            for text, value in options:
                ttk.Radiobutton(frame, text=text, value=value, variable=var,
                                command=self.on_settings).pack(side="left", padx=2)

        frame = ttk.LabelFrame(side, text="Alert threshold (5)", padding=4)
        frame.pack(anchor="w", fill="x", pady=(10, 0))
        self.thr_var = tk.DoubleVar()
        self.thr_text = tk.StringVar()
        ttk.Scale(frame, from_=0.0, to=0.5, variable=self.thr_var, command=self.on_threshold).pack(fill="x")
        row = ttk.Frame(frame)
        row.pack(fill="x", pady=2)
        entry = ttk.Entry(row, textvariable=self.thr_text, width=9)
        entry.pack(side="left")
        entry.bind("<Return>", self.on_threshold_typed)
        ttk.Button(row, text="Reset to default", command=self.reset_threshold).pack(side="right")
        self.thr_note = ttk.Label(frame, foreground=MUTED, wraplength=240)
        self.thr_note.pack(anchor="w")

    # ---------------------------------------------------------------- settings
    @property
    def pid(self) -> str:
        return self.patient_var.get().split(" — ")[0]

    @property
    def model(self) -> str:
        return self.model_var.get()

    @property
    def fset(self) -> str:
        return self.fset_var.get()

    @property
    def horizon(self) -> int:
        return int(self.horizon_var.get())

    @property
    def thr(self) -> float:
        return round(float(self.thr_var.get()), 4)

    def combo(self) -> tuple:
        return self.model, self.fset, self.horizon

    def default_threshold(self) -> float:
        # Floored (not rounded) to 4 decimals so the default never sits above the validated threshold.
        return float(np.floor(core.default_threshold(*self.combo()) * 1e4) / 1e4)

    def on_settings(self) -> None:
        """Model, feature set, horizon or patient changed: restore that combination's threshold and redraw."""
        self.set_threshold(self.thresholds.get(self.combo(), self.default_threshold()))
        val = core.default_threshold(*self.combo())
        self.thr_note.configure(text=f"Default = best-F1 threshold on the validation split ({val:.5f}).")
        self.schedule(self.current().refresh)

    def set_threshold(self, value: float) -> None:
        value = min(max(round(value, 4), 0.0), 0.5)
        self.thr_var.set(value)
        self.thr_text.set(f"{value:.4f}")
        self.thresholds[self.combo()] = value

    def on_threshold(self, _value=None) -> None:
        self.set_threshold(float(self.thr_var.get()))
        self.schedule(self.current().refresh)

    def on_threshold_typed(self, _event=None) -> None:
        try:
            self.set_threshold(float(self.thr_text.get()))
        except ValueError:
            self.thr_text.set(f"{self.thr:.4f}")
            return
        self.schedule(self.current().refresh)

    def reset_threshold(self) -> None:
        self.set_threshold(self.default_threshold())
        self.schedule(self.current().refresh)

    # ------------------------------------------------------------------ helpers
    def current(self):
        return self.tabs[self.notebook.index("current")]

    def schedule(self, func, delay_ms: int = 120) -> None:
        """Run func once the user pauses (slider drags fire many events)."""
        if self._pending:
            self.after_cancel(self._pending)
        self._pending = self.after(delay_ms, func)

    def show_missing(self, missing: list[str]) -> None:
        box = ttk.Frame(self, padding=30)
        box.pack(expand=True)
        ttk.Label(box, text="SepsisTRACE", font=("Helvetica", 22, "bold")).pack(anchor="w")
        ttk.Label(box, text="Some files the app needs are missing:", foreground="#C62828",
                  font=("Helvetica", 13)).pack(anchor="w", pady=8)
        ttk.Label(box, text="\n".join(missing), font=("Menlo", 12), justify="left").pack(anchor="w")
        ttk.Label(box, text="See README.md (section: Data and trained models).", foreground=MUTED).pack(
            anchor="w", pady=8)
