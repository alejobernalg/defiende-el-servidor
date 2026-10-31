#!/usr/bin/env bash
# Genera ../entrega_defiende_el_servidor.zip con el historial Git y la reconstrucción local de E4.
# Excluye entornos, cachés y datos (se recuperan con DVC).
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="$(cd .. && pwd)/entrega_defiende_el_servidor.zip"
rm -f "$OUT"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
git clone -q . "$TMP/proyecto"
git -C "$TMP/proyecto" remote set-url origin https://github.com/alejobernalg/defiende-el-servidor.git
MODEL_DIR="artifacts/e4_random_forest_reconstruido"
if [[ -f "$MODEL_DIR/model.joblib" && -f "$MODEL_DIR/reconstruction.json" ]]; then
  mkdir -p "$TMP/proyecto/$MODEL_DIR"
  cp "$MODEL_DIR/model.joblib" "$MODEL_DIR/reconstruction.json" "$TMP/proyecto/$MODEL_DIR/"
  [[ ! -f "$MODEL_DIR/val_predictions.csv" ]] || cp "$MODEL_DIR/val_predictions.csv" "$TMP/proyecto/$MODEL_DIR/"
else
  echo "Aviso: no se encontró la reconstrucción E4; el ZIP no incluirá model.joblib." >&2
fi
(cd "$TMP" && zip -qr "$OUT" proyecto -x "proyecto/.dvc/cache/*" "proyecto/.dvc/tmp/*")
echo "$OUT ($(du -h "$OUT" | cut -f1))"
