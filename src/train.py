"""Entrena una configuración y la registra en MLflow.

Uso: python src/train.py --config configs/e1_logreg_base.yaml

Ajusta todo el preprocesamiento solo con data/prepared/train.csv y evalúa en val.csv.
Registra parámetros, semilla, versión de datos (md5 DVC), commit Git, métricas de validación y el
pipeline completo (preprocesamiento + estimador) como artefacto.
"""
from __future__ import annotations

import argparse
import json
import platform
import time

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from common import (CATEGORICAL, PREP_DIR, ROOT, TARGET, binary_metrics, dvc_md5, feature_columns, git_commit,
                    git_is_clean, numeric_columns, sha256_file)

TRACKING_URI = f"sqlite:///{ROOT / 'mlflow.db'}"
EXPERIMENT = "defiende-el-servidor"
ARTIFACTS = ROOT / "artifacts"


def build_pipeline(cfg: dict, cat_cols: list[str], num_cols: list[str]) -> Pipeline:
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if cfg["preprocessing"].get("log1p_numeric"):
        num_steps.append(("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")))
    if cfg["model"] == "logistic_regression":
        num_steps.append(("scale", StandardScaler()))
    pre = ColumnTransformer([
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
        ("num", Pipeline(num_steps), num_cols),
    ])
    p, seed = cfg["params"], cfg["seed"]
    est = {
        "logistic_regression": lambda: LogisticRegression(random_state=seed, **p),
        "decision_tree": lambda: DecisionTreeClassifier(random_state=seed, **p),
        "random_forest": lambda: RandomForestClassifier(random_state=seed, **p),
    }[cfg["model"]]()
    return Pipeline([("preprocess", pre), ("model", est)])


def threshold_curve(y, score) -> pd.DataFrame:
    rows = [binary_metrics(y, score, t) for t in np.round(np.arange(0.01, 1.0, 0.01), 2)]
    return pd.DataFrame(rows).drop(columns=["zero_denominators"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--allow-dirty", action="store_true", help="permitir cambios sin commit (no recomendado)")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / args.config, encoding="utf-8"))
    if not git_is_clean() and not args.allow_dirty:
        raise SystemExit("Hay cambios sin commit: el run no sería trazable a un commit. Haz commit primero.")

    train = pd.read_csv(PREP_DIR / "train.csv")
    val = pd.read_csv(PREP_DIR / "val.csv")
    feats = feature_columns(train)
    cat_cols, num_cols = CATEGORICAL, numeric_columns(train)

    pipe = build_pipeline(cfg, cat_cols, num_cols)
    t0 = time.time()
    pipe.fit(train[feats], train[TARGET])
    fit_s = time.time() - t0
    score = pipe.predict_proba(val[feats])[:, 1]
    m = binary_metrics(val[TARGET], score, cfg["threshold"])
    m["roc_auc"] = float(roc_auc_score(val[TARGET], score))

    out = ARTIFACTS / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out / "model.joblib")
    preds = pd.DataFrame({"row_id": val.row_id, "y_true": val[TARGET], "score_attack": score,
                          "y_pred": (score >= cfg["threshold"]).astype(int)})
    preds.to_csv(out / "val_predictions.csv", index=False)
    threshold_curve(val[TARGET], score).to_csv(out / "threshold_curve.csv", index=False)
    cm = {"labels": ["0=normal", "1=ataque"], "matrix": [[m["tn"], m["fp"]], [m["fn"], m["tp"]]],
          "rows": "real", "cols": "predicho"}
    (out / "confusion_matrix.json").write_text(json.dumps(cm, indent=2))
    with open(out / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=cfg["name"]) as run:
        mlflow.set_tags({
            "exp_id": cfg["id"], "family": cfg["family"], "hypothesis": cfg["hypothesis"],
            "change": cfg["change"], "git_commit": git_commit(), "config_file": args.config,
            "dvc_raw_md5": dvc_md5("data/raw.dvc"), "dvc_prepared_md5": dvc_md5("data/prepared.dvc"),
            "train_sha256": sha256_file(PREP_DIR / "train.csv"), "val_sha256": sha256_file(PREP_DIR / "val.csv"),
            "positive_class": "1=ataque", "threshold_rule": "y_pred = 1 si score_attack >= threshold",
            "zero_division": "denominador cero -> 0.0, listado en zero_denominators",
            "python": platform.python_version(), "sklearn": sklearn.__version__,
        })
        mlflow.log_params({"model": cfg["model"], "seed": cfg["seed"], "threshold": cfg["threshold"],
                           "log1p_numeric": cfg["preprocessing"].get("log1p_numeric", False),
                           "n_train": len(train), "n_val": len(val), **{f"model__{k}": v for k, v in cfg["params"].items()}})
        mlflow.log_metrics({f"val_{k}": v for k, v in m.items() if isinstance(v, (int, float))} | {"fit_seconds": fit_s})
        mlflow.log_dict({"zero_denominators": m["zero_denominators"]}, "zero_denominators.json")
        for f in ("val_predictions.csv", "threshold_curve.csv", "confusion_matrix.json", "config.yaml"):
            mlflow.log_artifact(str(out / f))
        mlflow.sklearn.log_model(pipe, name="model", serialization_format="pickle",
                                 input_example=val[feats].head(5))
        (out / "run_id.txt").write_text(run.info.run_id)
    print(cfg["id"], run.info.run_id, {k: round(v, 4) for k, v in m.items() if isinstance(v, float)}, f"fit {fit_s:.1f}s")


if __name__ == "__main__":
    main()
