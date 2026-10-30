#!/usr/bin/env bash
# Prueba de recuperación en carpeta y entorno NUEVOS (sin cachés de la ejecución original).
#   bash scripts/recuperacion.sh <git-url> <commit> <sha-v1> <sha-v2> [carpeta-de-trabajo]
# Deja el log y las salidas en <carpeta>/repo/evidencias/recuperacion/entorno_nuevo/.
set -euo pipefail
GIT_URL=$1; COMMIT=$2; SHA_V1=$3; SHA_V2=$4
WORK=${5:-$(mktemp -d "${TMPDIR:-/tmp}/recuperacion.XXXXXX")}
mkdir -p "$WORK"; WORK=$(cd "$WORK" && pwd)
export HF_HOME="$WORK/hf_home"            # caché de Hugging Face vacía
export PIP_NO_CACHE_DIR=1
LOGTMP="$WORK/recuperacion.log"
exec > >(tee "$LOGTMP") 2>&1
set -x
date -u; uname -a; python3.12 --version
# 1. Código en la revisión elegida + dependencias fijadas
git clone "$GIT_URL" "$WORK/repo"
cd "$WORK/repo"
git checkout "$COMMIT"
git log -1 --format='%H %s'
python3.12 -m venv "$WORK/venv"
source "$WORK/venv/bin/activate"
pip install -q -r requirements.txt
pip freeze | grep -iE '^(scikit-learn|pandas|numpy|dvc|huggingface.hub|mlflow)=='
# 2. Datos desde el remote DVC entregado (caché .dvc/cache vacía en el clon nuevo)
dvc remote list
test ! -d .dvc/cache || ls .dvc/cache
dvc pull
dvc status
shasum -a 256 data/raw/*.csv data/prepared/*.csv
python - <<'PY'
import json, hashlib
def h(p): return hashlib.sha256(open(p,'rb').read()).hexdigest()
a = json.load(open('evidencias/auditoria.json')); m = json.load(open('data/prepared/manifest.json'))
checks = {f['file']: h(f['file']) == f['sha256'] == f['sha256_expected'] for f in a['files'].values()}
checks |= {s['file']: h(s['file']) == s['sha256'] for s in m['splits'].values()}
checks['data/prepared/splits.csv'] = h('data/prepared/splits.csv') == m['splits_csv_sha256']
print(json.dumps(checks, indent=1)); assert all(checks.values()), 'huellas no coinciden'
print('HUELLAS DE DATOS OK')
PY
# 3-5. Modelos desde Hugging Face por SHA completo, predicción y comparación
OUT=evidencias/recuperacion/entorno_nuevo
python src/recover.py --version v1 --revision "$SHA_V1" --cache-dir "$HF_HOME/hub" --out-dir "$OUT"
python src/recover.py --version v2 --revision "$SHA_V2" --cache-dir "$HF_HOME/hub" --out-dir "$OUT"
shasum -a 256 evidencias/recuperacion/*.csv "$OUT"/*.csv
# Métricas de test recalculables desde las predicciones entregadas
if [ -f evidencias/metricas-test.json ]; then python src/recompute_test_metrics.py; fi
set +x
echo "RECUPERACION OK en $WORK"
cp "$LOGTMP" "$OUT/recuperacion.log"
