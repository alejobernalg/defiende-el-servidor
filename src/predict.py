"""Carga una solución empaquetada (model.joblib + config.json) y predice.

Se copia tal cual en cada revisión de Hugging Face, por eso solo depende de pandas/joblib/scikit-learn.

Uso: python predict.py --model-dir <carpeta con model.joblib y config.json> --input datos.csv --output pred.csv
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_solution(model_dir):
    """Devuelve (pipeline, config). Verifica que el modelo sea el artefacto declarado en config.json."""
    model_dir = Path(model_dir)
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    actual = sha256_file(model_dir / "model.joblib")
    if actual != cfg["model_file_sha256"]:
        raise ValueError(f"model.joblib no coincide con config.json: {actual} != {cfg['model_file_sha256']}")
    return joblib.load(model_dir / "model.joblib"), cfg


def predict(pipe, cfg, df: pd.DataFrame) -> pd.DataFrame:
    """score_attack = P(ataque); y_pred = 1 si score_attack >= umbral (ataque = clase positiva)."""
    score = pipe.predict_proba(df[cfg["features"]])[:, 1]
    out = pd.DataFrame({"score_attack": score, "y_pred": (score >= cfg["threshold"]).astype(int)})
    if "row_id" in df:
        out.insert(0, "row_id", df["row_id"].values)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=".")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    pipe, cfg = load_solution(a.model_dir)
    predict(pipe, cfg, pd.read_csv(a.input, encoding="utf-8-sig")).to_csv(a.output, index=False)
    print(f"{cfg['version']} umbral={cfg['threshold']} -> {a.output}")


if __name__ == "__main__":
    main()
