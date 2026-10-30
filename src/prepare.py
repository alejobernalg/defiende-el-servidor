"""Prepara los datos: identificadores de fila, grupos de entradas idénticas y split train/val agrupado.

Política (ver README / auditoria.json):
- Solo se usa el archivo de entrenamiento publicado para crear train/val; test queda reservado.
- Se conservan todas las filas (los duplicados reflejan la frecuencia real del tráfico) y los grupos
  con etiquetas contradictorias (no se pueden "limpiar" sin usar la etiqueta; el error es irreducible).
- Las filas con las mismas variables predictoras forman un grupo y el grupo completo va a un solo lado
  (GroupShuffleSplit, 20 % de grupos a validación, semilla 42).
"""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from common import PREP_DIR, SEED, TARGET, TRAIN_FILE, feature_columns, read_raw, row_hash, sha256_file, write_json

VAL_FRACTION = 0.2


def main() -> None:
    df = read_raw(TRAIN_FILE)
    df.insert(0, "row_id", [f"train:{i}" for i in range(len(df))])  # i = fila de datos 0-based del CSV
    feats = feature_columns(df)
    df.insert(1, "group_id", row_hash(df, feats))

    gss = GroupShuffleSplit(n_splits=1, test_size=VAL_FRACTION, random_state=SEED)
    tr_idx, va_idx = next(gss.split(df, groups=df["group_id"]))
    df["split"] = "train"
    df.loc[df.index[va_idx], "split"] = "val"

    assert not set(df.loc[df.split == "train", "group_id"]) & set(df.loc[df.split == "val", "group_id"])

    PREP_DIR.mkdir(parents=True, exist_ok=True)
    df[["row_id", "id", "group_id", "split"]].to_csv(PREP_DIR / "splits.csv", index=False)
    cols = ["row_id", "id", "group_id"] + feats + ["attack_cat", TARGET]
    for name in ("train", "val"):
        df.loc[df.split == name, cols].to_csv(PREP_DIR / f"{name}.csv", index=False)

    manifest = {
        "source_file": TRAIN_FILE.name,
        "source_sha256": sha256_file(TRAIN_FILE),
        "seed": SEED,
        "method": "GroupShuffleSplit(test_size=0.2) sobre group_id = hash de las variables predictoras",
        "row_id_rule": "train:<i>, i = índice 0-based de la fila de datos en UNSW_NB15_training-set.csv",
        "features": feats,
        "splits": {},
    }
    for name in ("train", "val"):
        part = df[df.split == name]
        manifest["splits"][name] = {
            "rows": int(len(part)),
            "groups": int(part.group_id.nunique()),
            "class_counts": {str(k): int(v) for k, v in part[TARGET].value_counts().sort_index().items()},
            "file": f"data/prepared/{name}.csv",
            "sha256": sha256_file(PREP_DIR / f"{name}.csv"),
        }
    manifest["splits_csv_sha256"] = sha256_file(PREP_DIR / "splits.csv")
    write_json(PREP_DIR / "manifest.json", manifest)
    print({k: (v["rows"], v["class_counts"]) for k, v in manifest["splits"].items()})


if __name__ == "__main__":
    main()
