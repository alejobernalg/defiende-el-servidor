"""Utilidades compartidas: rutas, columnas, carga de datos, huellas y métricas."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PREP_DIR = ROOT / "data" / "prepared"
EVID_DIR = ROOT / "evidencias"

TRAIN_FILE = RAW_DIR / "UNSW_NB15_training-set.csv"
TEST_FILE = RAW_DIR / "UNSW_NB15_testing-set.csv"

# Huellas publicadas en el anexo de la guía (SHA-256 de los CSV sin comprimir).
EXPECTED_SHA256 = {
    "UNSW_NB15_training-set.csv": "bec7dd5ec88dc2a0ccc7a07879d338395ed7421750f675fd0339e07dfe0648fa",
    "UNSW_NB15_testing-set.csv": "734fe6642edf758f7c94d7d9149426b49d202fe8e7bf0bef47392489c3c0a559",
}
EXPECTED_ROWS = {"UNSW_NB15_training-set.csv": 175341, "UNSW_NB15_testing-set.csv": 82332}

TARGET = "label"
EXCLUDED = ["id", "attack_cat", "label"]  # id identifica filas; attack_cat revela la respuesta
CATEGORICAL = ["proto", "service", "state"]
SEED = 42


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_raw(path: Path) -> pd.DataFrame:
    # Los CSV traen BOM UTF-8 en la cabecera; utf-8-sig lo elimina.
    return pd.read_csv(path, encoding="utf-8-sig")


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in EXCLUDED and c not in ("row_id", "group_id", "split")]


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in feature_columns(df) if c not in CATEGORICAL]


def row_hash(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Hash estable por fila de las variables predictoras (sin id ni etiquetas)."""
    return pd.util.hash_pandas_object(df[cols], index=False).astype("uint64").map(lambda v: f"{v:016x}")


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def git_is_clean() -> bool:
    out = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True)
    return out.strip() == ""


def dvc_md5(dvc_file: str) -> str:
    """md5 del directorio registrado en un archivo .dvc (referencia de versión de datos)."""
    import yaml

    with open(ROOT / dvc_file) as f:
        return yaml.safe_load(f)["outs"][0]["md5"]


def safe_div(num: float, den: float) -> float:
    """Denominador cero -> 0.0 (y se reporta en las métricas con *_undefined)."""
    return float(num / den) if den else 0.0


def binary_metrics(y_true, score, threshold: float) -> dict:
    """Métricas con ataque (1) como clase positiva. Regla: y_pred = 1 si score >= umbral."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(score) >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    recall = safe_div(tp, tp + fn)
    precision = safe_div(tp, tp + fp)
    fpr = safe_div(fp, fp + tn)
    f1_pos = safe_div(2 * precision * recall, precision + recall)
    rec_neg, prec_neg = safe_div(tn, tn + fp), safe_div(tn, tn + fn)
    f1_neg = safe_div(2 * prec_neg * rec_neg, prec_neg + rec_neg)
    return {
        "threshold": float(threshold),
        "recall_attack": recall,
        "precision_attack": precision,
        "fpr": fpr,
        "f1_attack": f1_pos,
        "macro_f1": (f1_pos + f1_neg) / 2,
        "accuracy": safe_div(tp + tn, len(y_true)),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "n": int(len(y_true)),
        "zero_denominators": [k for k, d in
                              {"recall": tp + fn, "precision": tp + fp, "fpr": fp + tn}.items() if d == 0],
    }


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")
