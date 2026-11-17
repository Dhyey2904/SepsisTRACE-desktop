"""Patient explorer: time-travel (1), play, hide a measurement (4), contribution gauge (7) and what-if (6)."""
import tkinter as tk
from tkinter import ttk

import numpy as np

from backend import core
from backend.config import LABS, VITALS
from gui import charts
from gui.style import MUTED
from gui.widgets import Chart, Metric, MultiSelect, Tab, Table, fmt, hidden_curve, safe

PLAY_MS = 400            # one hour of the stay per 0.4 s while playing


class ExplorerTab(Tab):
    def __init__(self, parent, app):
        super().__init__(parent, app)
        self.shown_pid = None
        self.playing = False
        self.hide: list[str] = []
        self.hour = tk.IntVar(value=1)
        self.columnconfigure(0, weight=3)
        self.columnconfigure(1, weight=2)
        self.rowconfigure(3, weight=1)

        self.header = ttk.Label(self, font=("Helvetica", 15, "bold"))
        self.header.grid(row=0, column=0, columnspan=2, sticky="w")

        bar = ttk.Frame(self)
        bar.grid(row=1, column=0, columnspan=2, sticky="ew", pady=4)
        ttk.Label(bar, text="Time-travel (1): hour").pack(side="left")
        self.slider = tk.Scale(bar, from_=1, to=2, orient="horizontal", resolution=1, variable=self.hour,
                               command=lambda v: self.app.schedule(self.refresh), showvalue=True)
        self.slider.pack(side="left", fill="x", expand=True, padx=6)
        self.play_btn = ttk.Button(bar, text="▶ Play", width=9, command=self.toggle_play)
        self.play_btn.pack(side="left")

        # ---- left: hide picker, risk curve, metrics, measurement pattern
        left = ttk.Frame(self)
        left.grid(row=3, column=0, sticky="nsew", padx=(0, 10))
        left.columnconfigure(0, weight=1)
        left.rowconfigure(1, weight=3)
        left.rowconfigure(4, weight=2)
        hide_row = ttk.Frame(left)
        hide_row.grid(row=0, column=0, sticky="ew")
        ttk.Label(hide_row, text="Hide a measurement (4): treat as never measured, recompute live").pack(anchor="w")
        self.picker = MultiSelect(hide_row, VITALS + LABS, self.on_hide, height=3)
        self.picker.pack(side="left", fill="x", expand=True)
        ttk.Button(hide_row, text="Clear", command=self.picker.clear).pack(side="left", padx=4)

        self.risk_chart = Chart(left, 7, 2.8)
        self.risk_chart.grid(row=1, column=0, sticky="nsew", pady=4)
        tiles = ttk.Frame(left)
        tiles.grid(row=2, column=0, sticky="ew")
        self.m_risk, self.m_alert, self.m_hidden = (Metric(tiles, t) for t in
                                                    ("Risk now", "Alert", "Risk with hidden variables"))
        for m in (self.m_risk, self.m_alert, self.m_hidden):
            m.pack(side="left", expand=True, fill="x")
        ttk.Label(left, text="Measurement pattern (what model M sees)", font=("Helvetica", 11, "bold")).grid(
            row=3, column=0, sticky="w")
        self.mask_chart = Chart(left, 7, 2.6)
        self.mask_chart.grid(row=4, column=0, sticky="nsew")

        # ---- right: vitals, contribution gauge, what-if
        right = ttk.Frame(self)
        right.grid(row=3, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)
        ttk.Label(right, text="Vitals", font=("Helvetica", 11, "bold")).grid(row=0, column=0, sticky="w")
        self.m_note = ttk.Label(right, foreground="#1565C0", wraplength=420)
        self.vitals_chart = Chart(right, 5, 3)
        self.vitals_chart.grid(row=1, column=0, sticky="nsew")
        self.m_note.grid(row=2, column=0, sticky="w")

        ttk.Label(right, text="Contribution gauge (7): share of |SHAP| by feature block, XGBoost V+M",
                  font=("Helvetica", 11, "bold")).grid(row=3, column=0, sticky="w", pady=(6, 0))
        self.gauge_chart = Chart(right, 5, 0.75)
        self.gauge_chart.grid(row=4, column=0, sticky="ew")
        self.gauge_note = ttk.Label(right, foreground=MUTED)
        self.gauge_note.grid(row=5, column=0, sticky="w")
        self.top_table = Table(right, height=5)
        self.top_table.grid(row=6, column=0, sticky="ew")

        ttk.Label(right, text="What-if (6): change values at the current hour, then press Enter",
                  font=("Helvetica", 11, "bold")).grid(row=7, column=0, sticky="w", pady=(6, 0))
        form = ttk.Frame(right)
        form.grid(row=8, column=0, sticky="ew")
        self.whatif_vars, self.whatif_labels, self.whatif_defaults = {}, {}, {}
        for i, var in enumerate(core.WHATIF_VARS):
            self.whatif_labels[var] = ttk.Label(form, text=var)
            self.whatif_labels[var].grid(row=i // 2 * 2, column=i % 2, sticky="w", padx=4)
            self.whatif_vars[var] = tk.StringVar()
            entry = ttk.Entry(form, textvariable=self.whatif_vars[var], width=12)
            entry.grid(row=i // 2 * 2 + 1, column=i % 2, sticky="w", padx=4)
            entry.bind("<Return>", lambda e: self.update_whatif())
        ttk.Button(form, text="Recalculate", command=self.update_whatif).grid(row=0, column=2, rowspan=2, padx=8)
        ttk.Button(form, text="Reset values", command=self.reset_whatif).grid(row=2, column=2, rowspan=2, padx=8)
        self.m_raw = Metric(right, "Raw model score at this hour")
        self.m_raw.grid(row=9, column=0, sticky="w")
        self.whatif_note = ttk.Label(right, foreground=MUTED, wraplength=420)
        self.whatif_note.grid(row=10, column=0, sticky="w")

    # ------------------------------------------------------------------ events
    def on_hide(self, items: list[str]) -> None:
        self.hide = items
        self.app.schedule(self.refresh)

    def toggle_play(self) -> None:
        self.playing = not self.playing
        if self.playing and self.hour.get() >= self.max_hour:   # Play at the end replays from hour 1
            self.hour.set(1)
        self.play_btn.configure(text="⏸ Pause" if self.playing else "▶ Play")
        if self.playing:
            self.tick()

    def tick(self) -> None:
        if not self.playing:
            return
        self.refresh()
        if self.hour.get() >= self.max_hour:
            self.toggle_play()
            return
        self.hour.set(self.hour.get() + 1)
        self.after(PLAY_MS, self.tick)

    # ----------------------------------------------------------------- drawing
    @safe
    def refresh(self) -> None:
        a = self.app
        p = core.patient(a.pid)
        self.max_hour = int(p.hour.max())
        if a.pid != self.shown_pid:                 # new patient: start at the end of the stay, playback off
            self.shown_pid, self.playing = a.pid, False
            self.play_btn.configure(text="▶ Play")
            self.slider.configure(to=self.max_hour)
            self.hour.set(self.max_hour)
        hour = self.hour.get()
        onset = float(p.onset.iloc[0]) if p.is_septic.iloc[0] else None
        self.header.configure(text=f"Patient {a.pid} · {a.patients[a.pid]}"
                                   + (f" · onset at hour {onset:.0f}" if onset else ""))

        curves = {"risk": core.precomputed(a.pid, a.model, a.fset, a.horizon)}
        if self.hide:
            curves["risk with hidden variables"] = hidden_curve(a.pid, tuple(self.hide), a.model, a.fset, a.horizon)
        (ax,), = self.risk_chart.axes()
        charts.risk(ax, curves, hour, a.thr, onset)
        self.risk_chart.draw()

        now = curves["risk"].get(hour, np.nan)
        self.m_risk.set(fmt(now))
        self.m_alert.set("YES" if now >= a.thr else "no")
        if self.hide:
            h2 = curves["risk with hidden variables"].get(hour, np.nan)
            self.m_hidden.set(fmt(h2), h2 - now)
        else:
            self.m_hidden.set("—")

        greyed = a.fset == "M"
        self.m_note.configure(text="Feature set M: this model never sees values, only *when* things were measured. "
                                   "The vitals are greyed out because the model cannot use them." if greyed else "")
        axes = self.vitals_chart.axes(2, 2).flat
        charts.vitals(axes, p, hour, greyed)
        self.vitals_chart.draw()
        (ax,), = self.mask_chart.axes()
        charts.mask(ax, p, hour)
        self.mask_chart.draw()

        self.draw_gauge(a.pid, hour, a.horizon)
        self.fill_whatif(p, hour)
        self.update_whatif()

    def draw_gauge(self, pid: str, hour: int, horizon: int) -> None:
        c = core.contribution_for(pid, hour, "xgb", horizon)    # SHAP is exact for XGBoost
        (ax,), = self.gauge_chart.axes()
        if c is None:
            ax.axis("off")
            self.gauge_note.configure(text="No cached SHAP for this hour (predictions start at hour 3).")
            self.top_table.show(core.top_features_for(pid, hour, "xgb", horizon))
        else:
            charts.gauge(ax, [*c, max(0.0, 1 - sum(c))])
            self.gauge_note.configure(text=f"XGB V+M, {horizon}h. Top features this hour:")
            self.top_table.show(core.top_features_for(pid, hour, "xgb", horizon))
        self.gauge_chart.draw()

    # ------------------------------------------------------------------ what-if
    def fill_whatif(self, p, hour: int) -> None:
        """Prefill each box with the value now, else the last known value. Untouched boxes are not edits."""
        hist, cur = p[p.hour <= hour], p[p.hour == hour].iloc[0]
        self.whatif_patient, self.whatif_hour = p, hour
        for var in core.WHATIF_VARS:
            last = hist[var].ffill().iloc[-1]
            default = None if np.isnan(last) else round(float(last), 2)
            self.whatif_defaults[var] = default
            self.whatif_vars[var].set("" if default is None else f"{default:g}")
            self.whatif_labels[var].configure(text=f"{var} (now: {fmt(cur[var], '.2f')})")

    def reset_whatif(self) -> None:
        for var, default in self.whatif_defaults.items():
            self.whatif_vars[var].set("" if default is None else f"{default:g}")
        self.update_whatif()

    @safe
    def update_whatif(self) -> None:
        a, hour = self.app, self.whatif_hour
        edits = {}
        for var, sv in self.whatif_vars.items():
            text = sv.get().strip()
            if text and round(float(text), 2) != self.whatif_defaults[var]:
                edits[var] = float(text)
        if hour < 3:
            self.m_raw.set("—")
            self.whatif_note.configure(text="Predictions start at hour 3.")
            return
        before, after = (core.predict_with_edits(self.whatif_patient, e, hour, a.model, a.fset, a.horizon,
                                                 calibrated=False) for e in ({}, edits))
        self.m_raw.set(f"{after:.4f}", after - before if edits else None, f"{after - before:+.4f}")
        note = "Blank = not measured at this hour; typing a value there adds a measurement."
        if a.fset == "M" and edits:
            note = "Model M ignores values: only adding a measurement that was missing can move it. " + note
        self.whatif_note.configure(text=note)
