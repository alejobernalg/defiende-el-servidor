# Entrega — MLOps — Defiende el servidor

## Integrantes y repositorio
- Integrantes: Luna Lugo — Grupo B; Alejandro Bernal — Grupo B
- Asunto: `MLOps — Defiende el servidor — Grupo B — Luna Lugo, Alejandro Bernal`
- Repositorio de código: https://github.com/alejobernalg/defiende-el-servidor (público)
- Commit final de la entrega: el último commit de `main` (indicado en el mensaje de entrega). La prueba de recuperación se ejecutó sobre `b51c29ba1865cbab701223bd335650084c5dcf78`.

## Versiones
| Campo | v1 | v2 |
|---|---|---|
| Experimento | E1 `e1_logreg_base` (regresión logística) | E4 `e4_random_forest` (random forest 300 árboles) |
| Commit de entrenamiento | `ecbb4899184b23336917b40f32269aff490e89b5` | `ecbb4899184b23336917b40f32269aff490e89b5` |
| Tag de código | `code-v1` = `ecbb4899184b23336917b40f32269aff490e89b5` | `code-v2` = `43878a7f4a57fa283de43b24ba990a197a7b5598` (selección congelada en `f7d4ffa374445a8e126d764155f2a37b168f8f1d`) |
| Datos DVC (prepared) | `data/prepared.dvc` md5 `f6594eb80f87d21c69d1e10c46c0de5f.dir` (commit `8a7a81f4a0631049963f46a37c1614314de8dd46`) | igual |
| Datos DVC (raw) | `data/raw.dvc` md5 `fe93051ae209ffb0128a31e7a478c009.dir` (commit `119c57a299f4747316f766a6e081c2a318159db3`) | igual |
| Run ID MLflow | `ae051ab637994bb3bd29cf521f544087` | `ef3b2eaf90874ea5ae5b4b867327f00d` |
| Repositorio HF | https://huggingface.co/AlejandroLiberato/defiende-el-servidor-unsw-nb15 | igual |
| SHA revisión HF | `8033bee6b22195aa79e007085d680b104256a769` (tag `v1`) | `6282a4d6e05d473f37d0243f3bfa18b10116586e` (tag `v2`, revisión evaluada en test). Model card con el test: `bfdc6b94b54b1ed7febc7c45bd970c2b75d1e37a` (mismo `model.joblib`) |
| SHA-256 `model.joblib` | `2f8deb7a64a4607ccca04451e35919d73074f349e67828a4fe8b597547e845c0` | `03d398894717439d1e4bc20112309bf0a6a10a5727e2af4349e3baca2ad38a81` |
| Umbral | 0.50 | 0.64 |

## Evidencias
| Evidencia | Ubicación |
|---|---|
| Auditoría de datos y splits | `evidencias/auditoria.json`, `data/prepared/manifest.json` (vía DVC), `data/fuentes.json` |
| Tabla de experimentos | `evidencias/experimentos.csv` |
| Registros MLflow exportados | `evidencias/mlflow/` (`runs.csv`, `runs.json`, una carpeta por run ID) |
| Selección congelada | `evidencias/seleccion.json` (commit `f7d4ffa…`, anterior al test) |
| Test final | `evidencias/metricas-test.json`, `evidencias/predicciones-test.csv` (recalcular: `python src/recompute_test_metrics.py`) |
| Análisis de errores (posterior al test, descriptivo) | `evidencias/analisis-errores.json` |
| Recuperación | `evidencias/recuperacion/` (ver `LEEME.md`): resultado **OK**; v1 diferencia máx. 0.0, v2 2.2e-16 (tolerancia 1e-9); `y_pred` idéntico en ambas |
| Informe | `informe/informe.pdf` (fuente `informe/informe.html`) |

## Almacenamiento DVC
- Remote por defecto `hfstore`: https://huggingface.co/datasets/AlejandroLiberato/defiende-el-servidor-dvc-storage (dataset público de Hugging Face; `.dvc/config` usa `.../resolve/main` por HTTP; `dvc pull` sin credenciales).
- Si `dvc pull` falla con `CERTIFICATE_VERIFY_FAILED` (Python de python.org en macOS): `export SSL_CERT_FILE=$(python -m certifi)`.
- Acceso: los tres repositorios son públicos (GitHub, modelo y dataset de Hugging Face); no se requieren invitaciones ni tokens.

## Comandos
- Instalación: `python3.12 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
- Recuperación de datos: `dvc pull`
- Entrenamiento: `python src/train.py --config configs/e4_random_forest.yaml`
- Predicción: ver README («Descargar cada modelo y predecir»)
- Recuperación completa: `bash scripts/recuperacion.sh https://github.com/alejobernalg/defiende-el-servidor.git <commit> 8033bee6b22195aa79e007085d680b104256a769 6282a4d6e05d473f37d0243f3bfa18b10116586e`

## Tiempo y recursos (MacBook, 8 núcleos CPU, 16 GB RAM, sin GPU)
- Preparación + auditoría: ~15 s. E1: 4.8 s, E2: 6.5 s, E3: 4.5 s, E4: 54 s de ajuste.
- `model.joblib`: v1 12 KB, v2 163 MB. Evaluación de test: ~10 s.
- Prueba de recuperación completa en entorno nuevo: 745 s (la mayor parte es `pip install` sin caché).

## Limitaciones conocidas
- UNSW-NB15 es un banco de pruebas de 2015; no demuestra eficacia en una red actual.
- 10.4 % de las filas de test tienen entradas idénticas a filas de train/val.
- 229 grupos de entradas idénticas con etiquetas contradictorias en train (940 filas).
- El umbral de v2 se elige en la misma validación en la que se reportan sus métricas.
- En el mirror usado los archivos venían con los nombres intercambiados; se asignaron por SHA-256.



## Uso de IA, código externo y contribuciones
- Se usó Claude Code (Anthropic) para apoyar la escritura de scripts, la configuración de DVC/MLflow/Hugging Face y la documentación. También se usó Codex para contrastar esta entrega con la guía y revisar las evidencias pendientes. Las comprobaciones reproducibles guardadas incluyen la auditoría de fuentes y SHA-256, la asignación de grupos sin cruces entre train y validación, las métricas recalculables de test y la recuperación de v1/v2 sobre las mismas 25 filas de validación. Las predicciones recuperadas conservan `y_pred`; la diferencia máxima de puntuación fue 0.0 para v1 y 2.2e-16 para v2 (tolerancia 1e-9).
- No se adaptó código externo más allá de la documentación oficial de scikit-learn, MLflow, DVC y Hugging Face.
- Trazabilidad de commits: el historial de GitHub registra los commits de la entrega bajo la cuenta `alejobernalg` (Alejandro Bernal). Entre ellos están el commit `ecbb4899184b23336917b40f32269aff490e89b5` (cuatro experimentos), `b51c29ba1865cbab701223bd335650084c5dcf78` (análisis de errores y model card), `c62c2544eb6047ab026f86d42809e47b1a662cb4` (prueba de recuperación e informe) y `9ef6b14ec1964d526da92dee9ca500c831cfa28b` (script para generar el ZIP).
- Contribución de los integrantes: Luna Lugo y Alejandro Bernal trabajaron conjuntamente en el proyecto y comparten las decisiones, implementación, experimentos, documentación y verificación descritos en esta entrega. Alejandro publicó los commits desde su cuenta `alejobernalg`; por ello el historial identifica esa cuenta como autora de las publicaciones, aunque el trabajo se realizó en conjunto.
