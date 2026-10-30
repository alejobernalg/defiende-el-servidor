#!/usr/bin/env bash
# Genera ../entrega_defiende_el_servidor.zip con los archivos versionados en Git (incluye .git para el
# historial). Excluye entornos, cachés, datos (se recuperan con DVC) y modelos (se recuperan de HF).
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="$(cd .. && pwd)/entrega_defiende_el_servidor.zip"
rm -f "$OUT"
TMP=$(mktemp -d); git clone -q . "$TMP/proyecto"
git -C "$TMP/proyecto" remote set-url origin https://github.com/alejobernalg/defiende-el-servidor.git
(cd "$TMP" && zip -qr "$OUT" proyecto -x "proyecto/.dvc/cache/*" "proyecto/.dvc/tmp/*")
rm -rf "$TMP"
echo "$OUT ($(du -h "$OUT" | cut -f1))"
