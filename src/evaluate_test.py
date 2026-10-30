"""Evaluación ÚNICA de la solución congelada (v2) sobre todo UNSW_NB15_testing-set.csv.

Descarga v2 desde Hugging Face por su SHA completo (evidencias/seleccion.json), verifica el hash del
modelo contra la selección congelada y escribe evidencias/metricas-test.json y predicciones-test.csv.
Se niega a ejecutarse si ya existe metricas-test.json (no se re-evalúa ni se ajusta después).
"""
from __future__ import annotations

import json
import sys
import tempfile
from datetime import datetime, timezone

import pandas as pd
from huggingface_hub import snapshot_download

from common import EVID_DIR, TARGET, TEST_FILE, binary_metrics, dvc_md5, git_commit, read_raw, sha256_file, write_json
from predict import load_solution, predict


def main() -> None:
    out_json = EVID_DIR / "metricas-test.json"
    if out_json.exists():
        sys.exit("metricas-test.json ya existe: el test se evalúa una sola vez.")
    sel = json.loads((EVID_DIR / "seleccion.json").read_text())
    v = sel["chosen_version"]
    hf = sel["huggingface"][v]
    with tempfile.TemporaryDirectory() as cache:
        d = snapshot_download(hf["repo"], revision=hf["revision_sha"], cache_dir=cache)
        pipe, cfg = load_solution(d)
    assert cfg["model_file_sha256"] == sel[v]["model_file_sha256"], "el modelo descargado no es el congelado"
    assert cfg["threshold"] == sel[v]["threshold"]

    te = read_raw(TEST_FILE)
    te.insert(0, "row_id", [f"test:{i}" for i in range(len(te))])
    p = predict(pipe, cfg, te)
    preds = pd.DataFrame({"row_id": te.row_id, "id": te["id"], "y_true": te[TARGET],
                          "score_attack": p.score_attack, "y_pred": p.y_pred})
    preds.to_csv(EVID_DIR / "predicciones-test.csv", index=False, float_format="%.10f")
    # Se recalculan desde el CSV escrito para que cualquiera obtenga exactamente lo mismo.
    back = pd.read_csv(EVID_DIR / "predicciones-test.csv")
    m = binary_metrics(back.y_true, back.score_attack, cfg["threshold"])
    assert (back.y_pred == (back.score_attack >= cfg["threshold"]).astype(int)).all()

    write_json(out_json, {
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "version": v, "huggingface_repo": hf["repo"], "huggingface_revision_sha": hf["revision_sha"],
        "model_file_sha256": cfg["model_file_sha256"], "mlflow_run_id": cfg["mlflow_run_id"],
        "evaluation_code_commit": git_commit(),
        "dataset": {"file": "data/raw/UNSW_NB15_testing-set.csv", "sha256": sha256_file(TEST_FILE),
                    "dvc_raw_md5": dvc_md5("data/raw.dvc"), "rows_in_file": int(len(te))},
        "rows_evaluated": int(len(back)),
        "threshold": cfg["threshold"], "decision_rule": "y_pred = 1 si score_attack >= threshold",
        "positive_class": "1 = ataque",
        "zero_division_policy": "denominador cero -> 0.0 y se lista en zero_denominators",
        "confusion_matrix": {"rows": "real", "cols": "predicho", "labels": ["0=normal", "1=ataque"],
                             "matrix": [[m["tn"], m["fp"]], [m["fn"], m["tp"]]]},
        "metrics": {k: m[k] for k in ("recall_attack", "precision_attack", "fpr", "f1_attack", "macro_f1",
                                       "accuracy", "zero_denominators")},
        "validation_metrics_same_threshold": sel[v]["validation_metrics_at_threshold"],
        "recompute": "python src/recompute_test_metrics.py",
    })
    print(json.dumps(m, indent=1))


if __name__ == "__main__":
    main()
