"""Exporta los runs de MLflow a evidencias/ (consultable sin el servidor local).

- evidencias/experimentos.csv: una fila por configuración.
- evidencias/mlflow/runs.csv y runs.json: todos los parámetros, métricas y tags.
- evidencias/mlflow/<run_id>/: params.json, metrics.json, tags.json, meta.json y artefactos relevantes
  (config, matriz de confusión, curva de umbrales, predicciones de validación, MLmodel). Los pickles
  grandes (>5 MB) no se copian: el modelo está en MLflow (run ID) y en Hugging Face.
"""
from __future__ import annotations

import json
import shutil

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

from common import EVID_DIR, write_json
from train import EXPERIMENT, TRACKING_URI

CONCLUSIONS = {
    "E1": "Referencia funcional: recall alto pero FPR 0.17; demasiadas alertas sobre tráfico normal.",
    "E2": "log1p reduce FPR (0.174->0.134) y sube macro-F1 a costa de algo de recall: hipótesis confirmada en parte.",
    "E3": "El árbol mejora macro-F1 y FPR respecto a ambos lineales: las no linealidades importan.",
    "E4": "Mejor macro-F1, FPR y ROC-AUC de validación; puntuaciones más finas para ajustar umbral. Candidato a v2.",
}


def main() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()
    exp = client.get_experiment_by_name(EXPERIMENT)
    runs = client.search_runs([exp.experiment_id], order_by=["attributes.start_time ASC"])
    out = EVID_DIR / "mlflow"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    rows, full = [], []
    for r in runs:
        t, p, m = r.data.tags, r.data.params, r.data.metrics
        rid = r.info.run_id
        d = out / rid
        write_json(d / "params.json", p)
        write_json(d / "metrics.json", m)
        write_json(d / "tags.json", {k: v for k, v in t.items() if not k.startswith("mlflow.log-model")})
        write_json(d / "meta.json", {"run_id": rid, "run_name": r.info.run_name, "experiment": EXPERIMENT,
                                     "experiment_id": exp.experiment_id, "status": r.info.status,
                                     "start_time_ms": r.info.start_time, "end_time_ms": r.info.end_time,
                                     "logged_models": [lm.model_id for lm in client.search_logged_models(
                                         experiment_ids=[exp.experiment_id], filter_string=f"source_run_id='{rid}'")]})
        art = mlflow.artifacts.download_artifacts(run_id=rid, dst_path=str(d / "_tmp"))
        for f in ("config.yaml", "confusion_matrix.json", "threshold_curve.csv", "val_predictions.csv",
                  "zero_denominators.json"):
            shutil.copy(f"{art}/{f}", d / f)
        shutil.rmtree(d / "_tmp")
        lm = client.search_logged_models(experiment_ids=[exp.experiment_id], filter_string=f"source_run_id='{rid}'")[0]
        mdir = mlflow.artifacts.download_artifacts(artifact_uri=lm.artifact_location, dst_path=str(d / "_m"))
        (d / "model").mkdir()
        for f in ("MLmodel", "requirements.txt", "conda.yaml", "python_env.yaml", "model.pkl", "input_example.json"):
            src = f"{mdir}/{f}"
            try:
                if f == "model.pkl" and pd.io.common.file_exists(src) and __import__("os").path.getsize(src) > 5e6:
                    (d / "model" / "model.pkl.NOT_COPIED.txt").write_text(
                        "Pickle > 5 MB; recuperarlo desde MLflow (run ID) o Hugging Face (SHA en seleccion.json).\n")
                    continue
                shutil.copy(src, d / "model" / f)
            except FileNotFoundError:
                pass
        shutil.rmtree(d / "_m")

        rows.append({
            "exp_id": t["exp_id"], "name": r.info.run_name, "hypothesis": t["hypothesis"], "change": t["change"],
            "family": t["family"], "model": p["model"],
            "params": json.dumps({k: v for k, v in p.items() if k.startswith("model__") or k == "log1p_numeric"}),
            "config_file": t["config_file"], "seed": p["seed"], "git_commit": t["git_commit"],
            "dvc_raw_md5": t["dvc_raw_md5"], "dvc_prepared_md5": t["dvc_prepared_md5"], "run_id": rid,
            "threshold": float(p["threshold"]),
            "val_recall_attack": m["val_recall_attack"], "val_precision_attack": m["val_precision_attack"],
            "val_fpr": m["val_fpr"], "val_macro_f1": m["val_macro_f1"], "val_accuracy": m["val_accuracy"],
            "val_roc_auc": m["val_roc_auc"], "val_tp": int(m["val_tp"]), "val_fp": int(m["val_fp"]),
            "val_tn": int(m["val_tn"]), "val_fn": int(m["val_fn"]), "fit_seconds": round(m["fit_seconds"], 1),
            "conclusion": CONCLUSIONS[t["exp_id"]],
        })
        full.append({"run_id": rid, "run_name": r.info.run_name, "params": p, "metrics": m,
                     "tags": {k: v for k, v in t.items() if not k.startswith("mlflow.log-model")}})

    pd.DataFrame(rows).to_csv(EVID_DIR / "experimentos.csv", index=False)
    write_json(out / "runs.json", full)
    mlflow.search_runs([exp.experiment_id]).to_csv(out / "runs.csv", index=False)
    print(pd.DataFrame(rows)[["exp_id", "run_id", "val_recall_attack", "val_precision_attack", "val_fpr",
                              "val_macro_f1"]].to_string(index=False))


if __name__ == "__main__":
    main()
