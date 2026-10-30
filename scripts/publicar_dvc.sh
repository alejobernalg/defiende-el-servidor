#!/usr/bin/env bash
# Publica el almacenamiento DVC en un repositorio de datasets de Hugging Face (lectura pública por HTTP).
# Requiere `hf auth login`. El remote por defecto (hfstore) es HTTP de solo lectura; se sube vía una
# copia local del remote (staging) + `hf upload`, porque DVC no tiene un remote nativo de HF.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=$(python -c "import yaml;print(yaml.safe_load(open('configs/publicacion.yaml'))['hf_dvc_repo'])")
dvc remote add --local -f staging .dvc_storage
dvc push -r staging
hf repo create "$REPO" --repo-type dataset --exist-ok
hf upload "$REPO" .dvc_storage . --repo-type dataset --commit-message "DVC storage: datos raw y prepared"
dvc remote add -d -f hfstore "https://huggingface.co/datasets/$REPO/resolve/main"
echo "Remote DVC por defecto: $(dvc remote list)"
