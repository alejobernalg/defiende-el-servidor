"""Empaqueta v1/v2 y los publica en Hugging Face Hub (una revisión por versión + tags v1/v2).

  python src/publish_hf.py --build-only     # solo construye hf_build/v1, hf_build/v2 y las salidas de referencia
  python src/publish_hf.py                  # construye y publica (requiere `hf auth login`)

Cada revisión contiene: model.joblib (preprocesamiento + estimador), config.json (umbral, variables,
referencias), predict.py, requirements.txt y README.md (model card).
También genera evidencias/recuperacion/: 25 filas fijas de validación y las salidas de cada versión
calculadas con el artefacto original (antes de recuperar).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess

import pandas as pd
import sklearn
import yaml

from common import EVID_DIR, PREP_DIR, ROOT, TARGET, dvc_md5, feature_columns, sha256_file, write_json
from predict import load_solution, predict
from train import ARTIFACTS

PUB = yaml.safe_load(open(ROOT / "configs/publicacion.yaml", encoding="utf-8"))
BUILD = ROOT / "hf_build"
REC_DIR = EVID_DIR / "recuperacion"
N_RECOVERY_ROWS = 25

RUNTIME_REQS = ["scikit-learn==1.9.1", "numpy==2.5.4", "pandas==3.1.0", "scipy==1.18.1", "joblib==1.6.0"]


def git_rev(ref: str) -> str:
    return subprocess.check_output(["git", "rev-list", "-n", "1", ref], cwd=ROOT, text=True).strip()


def dvc_file_commit(path: str) -> str:
    return subprocess.check_output(["git", "log", "-n", "1", "--format=%H", "--", path], cwd=ROOT, text=True).strip()


def model_card(v: str, info: dict, cfg: dict, other: dict) -> str:
    m = info["validation_metrics_at_threshold"]
    gh = f"https://github.com/{PUB['github_repo']}"
    ex = pd.read_csv(EVID_DIR / "experimentos.csv").set_index("exp_id")
    diff = ""
    if v == "v2":
        o = other["validation_metrics_at_threshold"]
        diff = f"""
## Cambios respecto a v1

| Métrica (validación) | v1 (E1, umbral 0.5) | v2 ({info['experiment_id']}, umbral {info['threshold']}) |
|---|---|---|
| Recall de ataque | {o['recall_attack']:.4f} | {m['recall_attack']:.4f} |
| Precision de ataque | {o['precision_attack']:.4f} | {m['precision_attack']:.4f} |
| FPR | {o['fpr']:.4f} | {m['fpr']:.4f} |
| Macro-F1 | {o['macro_f1']:.4f} | {m['macro_f1']:.4f} |
| Falsos negativos (ataques perdidos) | {o['fn']} | {m['fn']} |
| Falsos positivos (alertas falsas) | {o['fp']} | {m['fp']} |

- **Qué cambió:** modelo (regresión logística → random forest de 300 árboles) y umbral (0.5 → {info['threshold']}).
- **Qué mejora:** la FPR cae a menos de un tercio y sube la precision: muchas menos alertas falsas.
- **Qué empeora:** el recall de ataque baja ({o['recall_attack']:.4f} → {m['recall_attack']:.4f}); se aceptan
  más ataques no detectados a cambio de reducir la carga de alertas. La regla fija recall ≥ 0.95.
- Razón: {info['reason']}
"""
    return f"""---
library_name: sklearn
tags: [tabular-classification, intrusion-detection, unsw-nb15, mlops-course]
license: other
---

# Defiende el servidor — UNSW-NB15 normal/ataque — **{v}**

Clasificador binario de conexiones de red: `0 = normal`, `1 = ataque` (clase positiva).
Trabajo académico del curso MLOps (Universidad Externado de Colombia, 2026-II). **No es un IDS de producción.**

## Propósito
Demostrar versionado y recuperación de código, datos, experimentos y modelos. La versión **{v}** es
{"la primera solución funcional (referencia lineal)." if v == "v1" else "la revisión elegida con la comparación de experimentos en validación."}

## Contenido de esta revisión
| Archivo | Descripción |
|---|---|
| `model.joblib` | `sklearn.Pipeline` completo: imputación + one-hot (proto, service, state) {"+ escalado " if cfg['model_family'] == 'logistic_regression' else ""}+ {cfg['estimator']} |
| `config.json` | Umbral, variables de entrada, regla de decisión, SHA-256 del modelo y referencias |
| `predict.py` | Carga y predicción (verifica el SHA-256 del modelo) |
| `requirements.txt` | Dependencias fijadas (Python {cfg['python']}) |

SHA-256 de `model.joblib`: `{info['model_file_sha256']}`

## Trazabilidad
| Pieza | Referencia |
|---|---|
| Código (GitHub) | [{gh}]({gh}) — commit de entrenamiento `{info['git_commit_training']}`; tag `{cfg['code_tag']}` = `{cfg['code_commit']}` |
| Datos (DVC) | `data/prepared.dvc` md5 `{info['dvc_prepared_md5']}` (commit `{cfg['dvc_prepared_commit']}`); `data/raw.dvc` md5 `{info['dvc_raw_md5']}` |
| Experimento (MLflow) | run `{info['mlflow_run_id']}` ({info['experiment_id']} `{info['experiment_name']}`), exportado en `evidencias/mlflow/` |
| Selección | `evidencias/seleccion.json` |

## Datos
UNSW-NB15, partición publicada `UNSW_NB15_training-set.csv` (175.341 filas, SHA-256 `bec7dd5e…48fa`),
dividida en train (140.145) / validación (35.196) con `GroupShuffleSplit` por grupos de entradas
idénticas (semilla 42; ningún grupo compartido). Variables: 42 predictoras (3 categóricas, 39 numéricas).
Excluidas: `id`, `attack_cat` (fuga de etiqueta) y `label`. Preprocesamiento ajustado solo con train.
`UNSW_NB15_testing-set.csv` (82.332 filas) reservado para la evaluación final de la versión elegida.

## Métricas de validación (umbral {info['threshold']})
| Recall ataque | Precision ataque | FPR | Macro-F1 | Accuracy |
|---|---|---|---|---|
| {m['recall_attack']:.4f} | {m['precision_attack']:.4f} | {m['fpr']:.4f} | {m['macro_f1']:.4f} | {m['accuracy']:.4f} |

Matriz de confusión (filas = real, columnas = predicho; 0 normal, 1 ataque): `[[TN={m['tn']}, FP={m['fp']}], [FN={m['fn']}, TP={m['tp']}]]`.
Denominadores cero → métrica 0.0 (no ocurrió en validación).
Hipótesis del experimento: {ex.loc[info['experiment_id'], 'hypothesis']}
{diff}
## Uso
```bash
pip install -r requirements.txt   # Python {cfg['python']}
```
```python
from huggingface_hub import snapshot_download
import pandas as pd, sys
d = snapshot_download("{PUB['hf_model_repo']}", revision="<SHA completo de la revisión {v}>")
sys.path.insert(0, d)
from predict import load_solution, predict
pipe, cfg = load_solution(d)
df = pd.read_csv("conexiones.csv")          # mismas columnas que UNSW-NB15 (sin necesidad de label)
print(predict(pipe, cfg, df))               # score_attack y y_pred (1 si score_attack >= {info['threshold']})
```

## Limitaciones
- Dataset sintético/histórico (2015, banco de pruebas de UNSW): no demuestra eficacia en una red actual.
- ~10 % de filas de test tienen entradas idénticas a filas de train/val (solapamiento de las particiones publicadas).
- 229 grupos de entradas idénticas tienen etiquetas contradictorias en train (940 filas): error irreducible.
- El umbral se eligió en la misma validación donde se reportan las métricas: pueden ser algo optimistas.
- El pickle de scikit-learn solo debe cargarse desde fuentes de confianza y con las versiones fijadas.

## Referencias
Moustafa, N., & Slay, J. (2015). UNSW-NB15: a comprehensive data set for network intrusion detection systems. MilCIS.
Moustafa, N., & Slay, J. (2016). The evaluation of NIDS datasets... Information Security Journal.
"""


def build() -> dict:
    sel = json.loads((EVID_DIR / "seleccion.json").read_text())
    val = pd.read_csv(PREP_DIR / "val.csv")
    feats = feature_columns(val)
    sample = val.sample(n=N_RECOVERY_ROWS, random_state=42).sort_values("row_id")
    REC_DIR.mkdir(parents=True, exist_ok=True)
    sample[["row_id"] + feats + [TARGET]].to_csv(REC_DIR / "entrada_validacion.csv", index=False)

    if BUILD.exists():
        shutil.rmtree(BUILD)
    built = {}
    for v in ("v1", "v2"):
        info = sel[v]
        d = BUILD / v
        d.mkdir(parents=True)
        shutil.copy(ARTIFACTS / info["experiment_name"] / "model.joblib", d / "model.joblib")
        assert sha256_file(d / "model.joblib") == info["model_file_sha256"]
        cfg = {
            "version": v, "threshold": info["threshold"], "decision_rule": info["threshold_rule"],
            "positive_class": 1, "classes": {"0": "normal", "1": "ataque"}, "features": feats,
            "excluded_columns": ["id", "attack_cat", "label"], "model_file": "model.joblib",
            "model_file_sha256": info["model_file_sha256"], "model_family": info["model_family"],
            "estimator": {"logistic_regression": "LogisticRegression", "tree": "RandomForestClassifier"}[info["model_family"]],
            "experiment_id": info["experiment_id"], "mlflow_run_id": info["mlflow_run_id"],
            "git_repo": f"https://github.com/{PUB['github_repo']}", "git_commit_training": info["git_commit_training"],
            "code_tag": f"code-{v}", "code_commit": git_rev(f"code-{v}"),
            "dvc_prepared_md5": info["dvc_prepared_md5"], "dvc_raw_md5": info["dvc_raw_md5"],
            "dvc_prepared_commit": dvc_file_commit("data/prepared.dvc"),
            "validation_metrics": info["validation_metrics_at_threshold"],
            "python": "3.12", "sklearn": sklearn.__version__,
        }
        write_json(d / "config.json", cfg)
        shutil.copy(ROOT / "src" / "predict.py", d / "predict.py")
        (d / "requirements.txt").write_text("\n".join(RUNTIME_REQS + ["huggingface_hub==2.4.0"]) + "\n")
        (d / "README.md").write_text(model_card(v, info, cfg, sel["v1"]), encoding="utf-8")

        pipe, c = load_solution(d)
        predict(pipe, c, sample).to_csv(REC_DIR / f"salida_{v}_original.csv", index=False)
        built[v] = d
    print("hf_build listo:", {v: sorted(p.name for p in d.iterdir()) for v, d in built.items()})
    return built


def publish(built: dict) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    repo = PUB["hf_model_repo"]
    api.create_repo(repo, repo_type="model", private=False, exist_ok=True)
    revs = {}
    for v in ("v1", "v2"):
        ci = api.upload_folder(repo_id=repo, folder_path=str(built[v]), commit_message=f"{v}: solución completa "
                               f"({json.loads((built[v] / 'config.json').read_text())['experiment_id']})")
        api.create_tag(repo, tag=v, revision=ci.oid, tag_message=f"Solución {v}", exist_ok=False)
        revs[v] = {"repo": repo, "url": f"https://huggingface.co/{repo}", "revision_sha": ci.oid, "tag": v,
                   "commit_url": ci.commit_url,
                   "model_file_sha256": json.loads((built[v] / "config.json").read_text())["model_file_sha256"]}
        print(v, ci.oid)
    sel = json.loads((EVID_DIR / "seleccion.json").read_text())
    sel["huggingface"] = revs
    write_json(EVID_DIR / "seleccion.json", sel)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-only", action="store_true")
    a = ap.parse_args()
    b = build()
    if not a.build_only:
        publish(b)
