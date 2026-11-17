"""Settings: loads config.yaml into a dict with attribute access (cfg.data.vitals)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]          # the python/ folder
CONFIG_FILE = ROOT / "config.yaml"

# The 40 hourly columns of a PhysioNet/CinC 2019 patient record.
VITALS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2"]
LABS = ["BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
        "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
        "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
        "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC",
        "Fibrinogen", "Platelets"]
STATICS = ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS"]
VARIABLE_COLUMNS = VITALS + LABS + STATICS


class Config(dict):
    """dict with attribute access, applied recursively (cfg.data.vitals)."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    @classmethod
    def wrap(cls, obj: Any) -> Any:
        if isinstance(obj, dict):
            return cls({k: cls.wrap(v) for k, v in obj.items()})
        if isinstance(obj, list):
            return [cls.wrap(v) for v in obj]
        return obj

    def path(self, key: str) -> Path:
        """Absolute path for cfg.paths[key], resolved against the python/ folder."""
        p = Path(self["paths"][key])
        return p if p.is_absolute() else ROOT / p


def load() -> Config:
    return Config.wrap(yaml.safe_load(CONFIG_FILE.read_text()))
