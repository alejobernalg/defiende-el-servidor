# Entrega — MLOps — Defiende el servidor

> Los campos marcados **PENDIENTE** se completan al publicar en GitHub/Hugging Face y al ejecutar la
> prueba de recuperación.

## Integrantes y repositorio
- Integrantes: Luna Lugo — Grupo B; Alejandro Bernal — Grupo B
- Asunto: `MLOps — Defiende el servidor — Grupo B — Luna Lugo, Alejandro Bernal`
- Repositorio de código: https://github.com/alejobernalg/defiende-el-servidor (privado; acceso de lectura para el docente por invitación)
- Commit final de la entrega: **PENDIENTE**

## Versiones
| Campo | v1 | v2 |
|---|---|---|
| Experimento | E1 `e1_logreg_base` (regresión logística) | E4 `e4_random_forest` (random forest 300 árboles) |
| Commit de entrenamiento | `ecbb4899184b23336917b40f32269aff490e89b5` | `ecbb4899184b23336917b40f32269aff490e89b5` |
| Tag de código | `code-v1` = `ecbb4899184b23336917b40f32269aff490e89b5` | `code-v2` = `43878a7f4a57fa283de43b24ba990a197a7b5598` (selección congelada en `f7d4ffa374445a8e126d764155f2a37b168f8f1d`) |
| Datos DVC (prepared) | `data/prepared.dvc` md5 `f6594eb80f87d21c69d1e10c46c0de5f.dir` (commit `8a7a81f4a0631049963f46a37c1614314de8dd46`) | igual |
| Datos DVC (raw) | `data/raw.dvc` md5 `fe93051ae209ffb0128a31e7a478c009.dir` (commit `119c57a299f4747316f766a6e081c2a318159db3`) | igual |
| Run ID MLflow | `ae051ab637994bb3bd29cf521f544087` | `ef3b2eaf90874ea5ae5b4b867327f00d` |
| Repositorio HF | **PENDIENTE** | **PENDIENTE** |
| SHA revisión HF | **PENDIENTE** (tag `v1`) | **PENDIENTE** (tag `v2`) |
| SHA-256 `model.joblib` | `2f8deb7a64a4607ccca04451e35919d73074f349e67828a4fe8b597547e845c0` | `03d398894717439d1e4bc20112309bf0a6a10a5727e2af4349e3baca2ad38a81` |
| Umbral | 0.50 | 0.64 |

## Evidencias
| Evidencia | Ubicación |
|---|---|
| Auditoría de datos y splits | `evidencias/auditoria.json`, `data/prepared/manifest.json` (vía DVC), `data/fuentes.json` |
| Tabla de experimentos | `evidencias/experimentos.csv` |
| Registros MLflow exportados | `evidencias/mlflow/` (`runs.csv`, `runs.json`, una carpeta por run ID) |
| Selección congelada | `evidencias/seleccion.json` (commit `f7d4ffa…`, anterior al test) |
| Test final | `evidencias/metricas-test.json`, `evidencias/predicciones-test.csv` |
| Recuperación | `evidencias/recuperacion/` (entrada fija de 25 filas, salidas originales, `entorno_nuevo/` con salidas recuperadas, comparaciones y log) |
| Informe | `informe.pdf` (adjunto aparte) |

## Almacenamiento DVC
- Remote por defecto `hfstore`: **PENDIENTE** (dataset de Hugging Face, lectura pública por HTTP; `dvc pull` sin credenciales).
- Respaldo: la carpeta `.dvc_storage` puede usarse como remote local: `dvc remote add -f -d respaldo <ruta>/.dvc_storage && dvc pull`.

## Comandos
- Instalación: `python3.12 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
- Recuperación de datos: `dvc pull`
- Entrenamiento: `python src/train.py --config configs/e4_random_forest.yaml`
- Predicción: ver README («Descargar cada modelo y predecir»)
- Recuperación completa: `bash scripts/recuperacion.sh <git-url> <commit> <sha-v1> <sha-v2>`

## Tiempo y recursos (MacBook, 8 núcleos CPU, 16 GB RAM, sin GPU)
- Preparación + auditoría: ~15 s. E1: 4.8 s, E2: 6.5 s, E3: 4.5 s, E4: 54 s de ajuste.
- `model.joblib`: v1 12 KB, v2 163 MB. Prueba de recuperación: **PENDIENTE**.

## Limitaciones conocidas
- UNSW-NB15 es un banco de pruebas de 2015; no demuestra eficacia en una red actual.
- 10.4 % de las filas de test tienen entradas idénticas a filas de train/val.
- 229 grupos de entradas idénticas con etiquetas contradictorias en train (940 filas).
- El umbral de v2 se elige en la misma validación en la que se reportan sus métricas.
- En el mirror usado los archivos venían con los nombres intercambiados; se asignaron por SHA-256.

## Uso de IA, código externo y contribuciones
- Se usó Claude Code (Anthropic) como asistente para escribir scripts, configurar DVC/MLflow/Hugging Face y
  redactar documentación. **PENDIENTE: describir qué verificaste tú** (p. ej., revisión de splits, métricas
  recalculadas, prueba de recuperación, lectura del código).
- No se adaptó código externo más allá de la documentación oficial de scikit-learn, MLflow, DVC y Hugging Face.
- Contribución de cada integrante: **PENDIENTE**.
