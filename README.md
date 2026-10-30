# Defiende el servidor — versionado de código, datos, experimentos y modelos

Clasificación binaria normal/ataque con **UNSW-NB15** (MLOps · 2026-II). El foco es la trazabilidad:

```
Código: commit Git → Datos: referencia DVC → Experimento: run ID MLflow → Modelo: revisión SHA de Hugging Face
```

| Pieza | Dónde |
|---|---|
| Código | este repositorio; tags `code-v1`, `code-v2` |
| Datos | `data/raw.dvc` (archivos recibidos), `data/prepared.dvc` (splits); remote DVC `hfstore` (dataset de Hugging Face, lectura HTTP) |
| Experimentos | MLflow local (`mlflow.db`), exportado en `evidencias/mlflow/` y `evidencias/experimentos.csv` |
| Modelos | Hugging Face Hub, revisiones `v1` y `v2` (ver `entrega.md` para los SHA completos) |

## Requisitos
- Python **3.12** (probado con 3.12 en macOS, CPU).
- Dependencias fijadas en `requirements.txt` (`scikit-learn==1.9.1`, `pandas==3.1.0`, `numpy==2.5.4`,
  `mlflow==3.17.2`, `dvc==3.67.1`, `huggingface_hub==2.4.0`, …).

## Instalación
```bash
git clone https://github.com/alejobernalg/defiende-el-servidor.git
cd defiende-el-servidor
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Recuperar los datos (DVC)
```bash
dvc pull                       # remote por defecto: hfstore (HTTP, no requiere credenciales)
shasum -a 256 data/raw/*.csv   # comparar con evidencias/auditoria.json
```
Versiones de datos:
- `data/raw.dvc` → `UNSW_NB15_training-set.csv` (175.341 filas) y `UNSW_NB15_testing-set.csv` (82.332 filas).
- `data/prepared.dvc` → `train.csv`, `val.csv`, `splits.csv`, `manifest.json`.

Para ver una versión concreta: `git checkout <commit> -- data/raw.dvc data/prepared.dvc && dvc checkout`.

## Reproducir la preparación y un experimento
```bash
cd src
python prepare.py        # regenera data/prepared/ (debe dar los mismos SHA-256 que manifest.json)
python audit.py          # regenera evidencias/auditoria.json
cd ..
python src/train.py --config configs/e1_logreg_base.yaml      # ~10 s
python src/train.py --config configs/e4_random_forest.yaml    # ~1 min en 8 núcleos
mlflow ui --backend-store-uri sqlite:///mlflow.db              # consultar runs en http://127.0.0.1:5000
python src/export_mlflow.py                                    # exporta a evidencias/
```
`train.py` se niega a correr con cambios sin commit, para que cada run quede asociado a un commit.

## Descargar cada modelo y predecir
```bash
# SHA completos en entrega.md / evidencias/seleccion.json
python - <<'PY'
import json, sys
from huggingface_hub import snapshot_download
import pandas as pd
sel = json.load(open("evidencias/seleccion.json"))["huggingface"]
for v in ("v1", "v2"):
    d = snapshot_download(sel[v]["repo"], revision=sel[v]["revision_sha"])
    sys.path.insert(0, d)
    from predict import load_solution, predict
    pipe, cfg = load_solution(d)                      # verifica el SHA-256 de model.joblib
    df = pd.read_csv("evidencias/recuperacion/entrada_validacion.csv")
    print(v, cfg["threshold"], predict(pipe, cfg, df).head())
    sys.path.pop(0); sys.modules.pop("predict")
PY
```
O con la línea de comandos incluida en cada revisión:
`python <carpeta-descargada>/predict.py --model-dir <carpeta-descargada> --input datos.csv --output pred.csv`.

## Prueba de recuperación en entorno nuevo
```bash
bash scripts/recuperacion.sh https://github.com/alejobernalg/defiende-el-servidor.git <commit> <sha-v1> <sha-v2>
```
Clona en una carpeta temporal, crea un venv nuevo, hace `dvc pull` sin caché, verifica huellas, descarga
v1 y v2 por SHA, predice las 25 filas fijas y compara con las salidas guardadas.

## Test final
```bash
python src/evaluate_test.py          # una sola vez (ya ejecutado; se niega si existe metricas-test.json)
python src/recompute_test_metrics.py # recalcula métricas desde evidencias/predicciones-test.csv
```

## Estructura
```
configs/      e1..e4 (experimentos), seleccion.yaml (regla pre-registrada), publicacion.yaml
src/          common, prepare, audit, train, export_mlflow, select_model, publish_hf, predict,
              evaluate_test, recompute_test_metrics, recover
scripts/      publicar_dvc.sh, recuperacion.sh
data/         raw.dvc, prepared.dvc, fuentes.json
evidencias/   auditoria.json, experimentos.csv, seleccion.json, metricas-test.json,
              predicciones-test.csv, mlflow/, recuperacion/
```

## Datos y atribución
UNSW-NB15 — Moustafa, N. & Slay, J. (2015), UNSW Canberra. Página oficial:
https://research.unsw.edu.au/projects/unsw-nb15-dataset. Uso académico.
