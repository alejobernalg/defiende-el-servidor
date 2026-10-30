"""Análisis descriptivo de errores de v2 en validación y test (posterior al test; no cambia modelo ni umbral).

-> evidencias/analisis-errores.json
"""
from __future__ import annotations

import json

import pandas as pd

from common import EVID_DIR, PREP_DIR, TEST_FILE, read_raw, write_json
from train import ARTIFACTS


def by(df, col, mask_label):
    d = df[df.label == mask_label]
    g = d.groupby(col)["y_pred"].agg(["size", "mean"]).sort_values("size", ascending=False)
    key = "fpr" if mask_label == 0 else "recall_attack"
    return {str(k): {"rows": int(r["size"]), key: round(float(r["mean"]), 4)} for k, r in g.iterrows()}


def main() -> None:
    sel = json.loads((EVID_DIR / "seleccion.json").read_text())
    thr = sel["v2"]["threshold"]
    val = pd.read_csv(PREP_DIR / "val.csv")
    vp = pd.read_csv(ARTIFACTS / sel["v2"]["experiment_name"] / "val_predictions.csv")
    val["y_pred"] = (vp.score_attack >= thr).astype(int).values
    te = read_raw(TEST_FILE)
    te["y_pred"] = pd.read_csv(EVID_DIR / "predicciones-test.csv").y_pred.values
    out = {"version": "v2", "threshold": thr,
           "note": "Descriptivo, calculado después de la evaluación final; no se usó para ajustar nada."}
    for name, d in (("validation", val), ("test", te)):
        out[name] = {"rows": int(len(d)), "normal_share": round(float((d.label == 0).mean()), 4),
                     "fpr_by_service": by(d, "service", 0), "fpr_by_state": by(d, "state", 0),
                     "recall_by_attack_cat": by(d, "attack_cat", 1)}
    write_json(EVID_DIR / "analisis-errores.json", out)
    print(json.dumps({k: out[k]["fpr_by_service"]["-"] for k in ("validation", "test")}))


if __name__ == "__main__":
    main()
