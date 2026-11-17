# SepsisTRACE: desktop app (Tkinter)

A desktop app that replays real ICU stays hour by hour and shows how machine-learning models
score each patient's risk of sepsis. The question behind it: **can the *pattern* of clinical
measurement (what was measured, and when) predict sepsis, independently of the values?**

Data: PhysioNet/CinC Challenge 2019. Every patient shown comes from the held-out test split.

## How to run

```bash
git clone https://github.com/Dhyey2904/SepsisTRACE-desktop.git
cd SepsisTRACE-desktop
pip install -r requirements.txt      # macOS: also `brew install libomp` for xgboost
python main.py
```

## Folder structure

```
SepsisTRACE-desktop/
├── main.py                 entry point: opens the window
├── config.yaml             settings: file paths, variables, models, horizons
├── requirements.txt        libraries needed
│
├── gui/                    user interface (Tkinter + matplotlib)
│   ├── app.py              main window and sidebar (patient, model, feature set, horizon, threshold)
│   ├── widgets.py          reusable widgets: Chart, Table, Metric tile, MultiSelect, Tab base class
│   ├── charts.py           drawings: risk curve, vitals, measurement pattern, SHAP gauge
│   ├── style.py            colours and chart style
│   └── tabs/
│       ├── explorer.py     Patient explorer tab
│       ├── console.py      Threshold console tab
│       └── challenge.py    Clinician challenge tab
│
├── backend/                logic, no GUI code
│   ├── config.py           reads config.yaml; the 40 patient variables
│   ├── features.py         turns a patient's hourly record into model features (V / M / VM)
│   ├── models.py           loads trained models and calibrators; clinical alert metrics
│   └── core.py             everything the GUI asks for: patients, predictions, SHAP, threshold metrics
│
├── data/                   demo data (30 test patients + test-split predictions)
└── trained_models/         48 trained models (4 models × 3 feature sets × 4 horizons) + calibrators
```

The GUI only talks to `backend/core.py`; the backend never imports anything from the GUI.

## What the app does

The left sidebar picks the patient, **model (2)** (`lr` logistic regression, `rf` random forest,
`xgb` XGBoost, `mlp` neural network), **feature set (3)**, prediction horizon (1, 3, 6 or 12 hours
ahead) and the **alert threshold (5)**.

| Feature set | What the model sees |
|---|---|
| **V** | the measured values (vitals and labs) over the last 12 hours |
| **M** | only *when* things were measured, never the values |
| **VM** | both |

All three also include six static fields (age, gender, ICU unit, ...).

| Tab | Features |
|---|---|
| **Patient explorer** | **(1) Time-travel**: slide or play through the stay hour by hour. **(4) Hide a measurement**: treat chosen variables as never measured and recompute the risk live. **(6) What-if**: change HR, MAP, Temp or Lactate at the current hour and see the new score. **(7) Contribution gauge**: how much of the XGBoost decision came from values vs. measurement pattern (SHAP). |
| **Threshold console (5)** | Applies the threshold to the whole test split: share of septic patients alerted before onset, sensitivity, false alarms per patient-day, median lead time, confusion matrix. |
| **Clinician challenge (8)** | Shows a random patient's record; guess whether sepsis is coming, then reveal the truth and what each model predicted. |

## Data and trained models

| File | Contents |
|---|---|
| `data/demo_patients.parquet` | hourly records of the 30 demo patients |
| `data/demo_predictions.parquet` | stored risk curves for every model / feature set / horizon |
| `data/shap_cache.parquet` | SHAP explanations for XGBoost V+M |
| `data/test_predictions.parquet` | predictions for the whole test split (used by the threshold console) |
| `data/demo_thresholds.json` | default threshold per model (best F1 on the validation split) |
| `data/test_onsets.json` | sepsis onset hour for each septic test patient |
| `trained_models/*.joblib` | fitted model pipelines (`<model>_<set>_<horizon>h_s42.joblib`, compressed) and their probability calibrators (`.calib.joblib`) |

If any of these are missing, the app opens a screen listing them instead of crashing.
