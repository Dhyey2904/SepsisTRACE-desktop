"""Threshold console (5): the alert threshold applied to the whole internal test split."""
from tkinter import ttk

import numpy as np
import pandas as pd

from gui.widgets import Metric, Tab, Table, console, fmt, safe
from gui.style import MUTED


class ConsoleTab(Tab):
    def __init__(self, parent, app):
        super().__init__(parent, app)
        self.header = ttk.Label(self, font=("Helvetica", 15, "bold"))
        self.header.pack(anchor="w")
        ttk.Label(self, text="Move the alert threshold in the sidebar: every number below updates.",
                  foreground=MUTED).pack(anchor="w", pady=(0, 8))
        tiles = ttk.Frame(self)
        tiles.pack(fill="x")
        self.detected, self.sens, self.fa, self.lead = (Metric(tiles, t) for t in (
            "Septic patients alerted before onset", "Hour-level sensitivity", "False alarms / patient-day",
            "Median lead time (detected)"))
        for m in (self.detected, self.sens, self.fa, self.lead):
            m.pack(side="left", expand=True, fill="x")
        self.warning = ttk.Label(self, foreground="#B26A00", wraplength=900)
        self.warning.pack(anchor="w", pady=4)
        ttk.Label(self, text="Hour-level confusion matrix", font=("Helvetica", 11, "bold")).pack(anchor="w", pady=(10, 2))
        self.table = Table(self, height=2)
        self.table.pack(anchor="w")
        self.extra = Table(self, height=7)
        ttk.Label(self, text="All metrics at this threshold", font=("Helvetica", 11, "bold")).pack(anchor="w", pady=(14, 2))
        self.extra.pack(anchor="w", fill="x")
        ttk.Label(self, text="Lead time is measured over detected septic patients only; read it together with the "
                             "detected fraction.", foreground=MUTED).pack(anchor="w", pady=6)

    @safe
    def refresh(self) -> None:
        a = self.app
        self.header.configure(text=f"Threshold console (5): whole internal test split, "
                                   f"{a.model} · {a.fset} · {a.horizon}h · threshold {a.thr:.4f}")
        r = console(a.model, a.fset, a.horizon, a.thr)
        self.detected.set(fmt(r["patients_detected_fraction"], ".0%"))
        self.sens.set(f"{r['hour_sensitivity']:.0%}")
        self.fa.set(fmt(r["false_alarms_per_patient_day"], ".2f"))
        lead = r["median_lead_time_hours"]
        self.lead.set("—" if np.isnan(lead) else f"{lead:.0f} h")
        self.warning.configure(text="" if r["onsets_available"] else
                               "Onset hours not found (data/test_onsets.json): patient-level metrics are hidden.")
        cm = r["confusion"]
        self.table.show(pd.DataFrame({"": [f"label y_{a.horizon}=1", f"label y_{a.horizon}=0"],
                                      "alert": [cm["TP"], cm["FP"]], "no alert": [cm["FN"], cm["TN"]]}))
        rows = {k: v for k, v in r.items() if isinstance(v, float)}
        self.extra.show(pd.DataFrame({"metric": list(rows), "value": list(rows.values())}), digits=4)
