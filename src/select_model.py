"""Aplica la regla pre-registrada (configs/seleccion.yaml) usando SOLO validación.

Escribe evidencias/seleccion.json (sin referencias de Hugging Face; las añade publish_hf.py).
El commit que contiene este archivo es la evidencia de congelación anterior a la evaluación en test.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import yaml

from common import EVID_DIR, ROOT, dvc_md5, git_commit, sha256_file, write_json
from train import ARTIFACTS

V1_EXPERIMENT = "e1_logreg_base"  # v1 = primera solución funcional: la referencia, umbral 0.5


def main() -> None:
    rule = yaml.safe_load(open(ROOT / "configs/seleccion.yaml", encoding="utf-8"))
    exp = pd.read_csv(EVID_DIR / "experimentos.csv")
    best = exp.sort_values(["val_macro_f1", "val_fpr"], ascending=[False, True]).iloc[0]

    curve = pd.read_csv(ARTIFACTS / best["name"] / "threshold_curve.csv")
    ok = curve[curve.recall_attack >= rule["threshold_rule"]["min_recall_attack"]]
    if len(ok):
        chosen = ok.sort_values(["fpr", "threshold"], ascending=[True, False]).iloc[0]
        why_thr = (f"Mínima FPR con recall de ataque >= {rule['threshold_rule']['min_recall_attack']} en validación "
                   f"(rejilla 0.01-0.99).")
    else:
        chosen = curve.sort_values("macro_f1", ascending=False).iloc[0]
        why_thr = "Ningún umbral alcanza el recall mínimo; se usa el de mayor macro-F1."
    thr = float(round(chosen.threshold, 2))
    at05 = curve[curve.threshold.round(2) == 0.5].iloc[0]

    def version(name, threshold, reason):
        row = exp[exp.name == name].iloc[0]
        c = pd.read_csv(ARTIFACTS / name / "threshold_curve.csv")
        mm = c[c.threshold.round(2) == round(threshold, 2)].iloc[0]
        return {
            "experiment_id": row["exp_id"], "experiment_name": name, "mlflow_run_id": row["run_id"],
            "git_commit_training": row["git_commit"], "dvc_prepared_md5": row["dvc_prepared_md5"],
            "dvc_raw_md5": row["dvc_raw_md5"], "model_family": row["family"],
            "model_file_sha256": sha256_file(ARTIFACTS / name / "model.joblib"),
            "threshold": threshold, "threshold_rule": "y_pred = 1 si score_attack >= threshold",
            "validation_metrics_at_threshold": {k: (float(mm[k]) if k not in "tp fp tn fn n" else int(mm[k]))
                                                for k in ["recall_attack", "precision_attack", "fpr", "macro_f1",
                                                          "accuracy", "tp", "fp", "tn", "fn", "n"]},
            "reason": reason,
        }

    sel = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "selection_code_parent_commit": git_commit(),
        "freeze_evidence": "El commit de Git que añade este archivo (tag code-v2) es anterior a cualquier "
                           "evaluación en test; ver git log de evidencias/metricas-test.json.",
        "rule": rule,
        "chosen_version": "v2",
        "v1": version(V1_EXPERIMENT, 0.5, "Primera solución funcional: referencia lineal con umbral por defecto 0.5."),
        "v2": version(best["name"], thr,
                      f"Modelo: {best['exp_id']} tiene el mayor macro-F1 de validación con umbral 0.5 "
                      f"({best['val_macro_f1']:.4f}). Umbral: {why_thr} En 0.5 tenía recall "
                      f"{at05.recall_attack:.4f} y FPR {at05.fpr:.4f}; con {thr} recall {chosen.recall_attack:.4f} "
                      f"y FPR {chosen.fpr:.4f}."),
        "test_used_for_selection": False,
        "huggingface": None,
    }
    write_json(EVID_DIR / "seleccion.json", sel)
    print(sel["v2"]["reason"])


if __name__ == "__main__":
    main()
