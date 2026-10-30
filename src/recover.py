"""Prueba de recuperación de una versión desde Hugging Face por SHA completo.

python src/recover.py --version v1 --revision <SHA> --out-dir <carpeta>

Descarga la revisión en una caché nueva (--cache-dir), verifica el SHA-256 del modelo, predice sobre
evidencias/recuperacion/entrada_validacion.csv y compara con salida_<v>_original.csv:
y_pred debe ser idéntico y |score_attack| con diferencia máxima <= TOLERANCE.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
REC = ROOT / "evidencias" / "recuperacion"
TOLERANCE = 1e-9


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True, choices=["v1", "v2"])
    ap.add_argument("--revision", required=True, help="SHA completo (40 hex) de la revisión de Hugging Face")
    ap.add_argument("--repo", default=json.loads((ROOT / "evidencias/seleccion.json").read_text())["huggingface"]["v1"]["repo"])
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    assert len(a.revision) == 40, "usar el SHA completo, no un tag"

    d = snapshot_download(a.repo, revision=a.revision, cache_dir=a.cache_dir)
    sys.path.insert(0, d)  # predict.py de la propia revisión descargada
    from predict import load_solution, predict, sha256_file

    pipe, cfg = load_solution(d)
    inp = pd.read_csv(REC / "entrada_validacion.csv")
    got = predict(pipe, cfg, inp)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    got.to_csv(out / f"salida_{a.version}_recuperada.csv", index=False)

    exp = pd.read_csv(REC / f"salida_{a.version}_original.csv")
    max_diff = float((got.score_attack - exp.score_attack).abs().max())
    res = {
        "version": a.version, "repo": a.repo, "revision_sha": a.revision, "downloaded_to": d,
        "files": {f.name: sha256_file(f) for f in sorted(Path(d).iterdir()) if f.is_file()},
        "config_version": cfg["version"], "threshold": cfg["threshold"], "mlflow_run_id": cfg["mlflow_run_id"],
        "model_file_sha256": cfg["model_file_sha256"], "rows": int(len(inp)),
        "row_ids_equal": bool((got.row_id == exp.row_id).all()),
        "y_pred_equal": bool((got.y_pred == exp.y_pred).all()),
        "score_max_abs_diff": max_diff, "tolerance": TOLERANCE,
        "y_pred_counts": {str(k): int(v) for k, v in got.y_pred.value_counts().sort_index().items()},
    }
    res["passed"] = res["row_ids_equal"] and res["y_pred_equal"] and max_diff <= TOLERANCE
    (out / f"comparacion_{a.version}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    sys.exit(0 if res["passed"] else 1)


if __name__ == "__main__":
    main()
