"""Clinician challenge (8): look at a hidden test patient and guess whether sepsis is coming."""
import tkinter as tk
from tkinter import ttk

import numpy as np
import pandas as pd

from backend import core
from gui import charts
from gui.style import MUTED
from gui.widgets import Chart, Metric, Tab, Table, safe

GUESSES = ["Sepsis coming", "No sepsis"]


class ChallengeTab(Tab):
    def __init__(self, parent, app):
        super().__init__(parent, app)
        self.pid = None
        self.revealed = False
        self.score = self.rounds = 0
        self.guess = tk.StringVar(value=GUESSES[0])

        ttk.Label(self, text="Clinician challenge (8)", font=("Helvetica", 15, "bold")).pack(anchor="w")
        ttk.Label(self, text="A hidden test patient. Look at the record and decide whether they develop sepsis.",
                  foreground=MUTED).pack(anchor="w")
        self.vitals_chart = Chart(self, 11, 2.2)
        self.vitals_chart.pack(fill="both", expand=True, pady=4)
        self.mask_chart = Chart(self, 11, 2.4)
        self.mask_chart.pack(fill="both", expand=True)

        row = ttk.Frame(self)
        row.pack(fill="x", pady=6)
        ttk.Label(row, text="Your call:").pack(side="left")
        self.radios = [ttk.Radiobutton(row, text=g, value=g, variable=self.guess) for g in GUESSES]
        for r in self.radios:
            r.pack(side="left", padx=6)
        self.reveal_btn = ttk.Button(row, text="Reveal", command=self.reveal)
        self.reveal_btn.pack(side="left", padx=(20, 6))
        ttk.Button(row, text="Next patient", command=self.next_patient).pack(side="left")
        self.m_score = Metric(row, "Session score")
        self.m_score.pack(side="right")

        self.truth = ttk.Label(self, font=("Helvetica", 12, "bold"))
        self.truth.pack(anchor="w")
        self.table = Table(self, height=4)
        self.next_patient(draw=False)

    def next_patient(self, draw: bool = True) -> None:
        """Draw a different patient from the one on screen."""
        others = self.app.patients.index.difference([self.pid])
        self.pid = str(np.random.default_rng().choice(others))
        self.revealed = False
        if draw:
            self.refresh()

    def reveal(self) -> None:
        if self.revealed:
            return
        truth = bool(core.patient(self.pid).is_septic.iloc[0])
        self.revealed = True
        self.rounds += 1
        self.score += int((self.guess.get() == "Sepsis coming") == truth)
        self.refresh()

    @safe
    def refresh(self) -> None:
        cp = core.patient(self.pid)
        last = int(cp.hour.max())
        charts.vitals(self.vitals_chart.axes(1, 4).flat, cp, last, greyed=False)
        self.vitals_chart.draw()
        (ax,), = self.mask_chart.axes()
        charts.mask(ax, cp, last)
        self.mask_chart.draw()

        state = "disabled" if self.revealed else "normal"
        self.reveal_btn.configure(state=state)
        for r in self.radios:
            r.configure(state=state)
        self.m_score.set(f"{self.score} / {self.rounds}")
        if not self.revealed:
            self.truth.configure(text="")
            self.table.pack_forget()
            return
        truth = bool(cp.is_septic.iloc[0])
        self.truth.configure(text="Truth: " + (f"septic, onset at hour {cp.onset.iloc[0]:.0f}" if truth else "not septic"))
        h, rows = self.app.horizon, []
        for m in core.MODELS:
            s = core.precomputed(self.pid, m, "VM", h)
            first = charts.first_alert(s, core.default_threshold(m, "VM", h))
            rows.append({"model": m, "max risk": charts.max_or_nan(s), "alerted": first is not None,
                         "first alert hour": first})
        self.table.show(pd.DataFrame(rows))
        self.table.pack(anchor="w", fill="x")
