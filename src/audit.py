"""Auditoría de los archivos recibidos y de las particiones -> evidencias/auditoria.json.

Del archivo de test solo se auditan esquema, duplicados y cruces de entradas; sus etiquetas no se usan
para ninguna decisión de modelado.
"""
from __future__ import annotations

import json
from datetime import date

import pandas as pd

from common import (CATEGORICAL, EVID_DIR, EXCLUDED, EXPECTED_ROWS, EXPECTED_SHA256, PREP_DIR, SEED, TARGET,
                    TEST_FILE, TRAIN_FILE, feature_columns, read_raw, row_hash, sha256_file, write_json)

with open(PREP_DIR.parents[0] / "fuentes.json", encoding="utf-8") as f:
    SOURCES = json.load(f)


def file_report(path, df, feats):
    sha = sha256_file(path)
    groups = row_hash(df, feats)
    dup_rows = int(groups.duplicated(keep="first").sum())
    lab = df.groupby(groups)[TARGET].nunique()
    contradictory = lab[lab > 1].index
    in_contra = groups.isin(contradictory)
    return {
        "file": f"data/raw/{path.name}",
        "source": SOURCES["files"][path.name],
        "sha256": sha,
        "sha256_expected": EXPECTED_SHA256[path.name],
        "sha256_matches_annex": sha == EXPECTED_SHA256[path.name],
        "bytes": path.stat().st_size,
        "rows": int(len(df)),
        "rows_expected": EXPECTED_ROWS[path.name],
        "columns": int(df.shape[1]),
        "column_names": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "class_counts_label": {str(k): int(v) for k, v in df[TARGET].value_counts().sort_index().items()},
        "attack_cat_counts": {str(k): int(v) for k, v in df["attack_cat"].value_counts().items()},
        "missing_values_total": int(df.isna().sum().sum()),
        "missing_by_column": {c: int(v) for c, v in df.isna().sum().items() if v},
        "id_unique": bool(df["id"].is_unique),
        "duplicates_by_features": {
            "unique_feature_groups": int(groups.nunique()),
            "duplicate_rows_beyond_first": dup_rows,
            "rows_in_groups_size_gt1": int(groups.duplicated(keep=False).sum()),
        },
        "contradictory_label_groups": {
            "groups": int(len(contradictory)),
            "rows": int(in_contra.sum()),
        },
    }, groups


def main() -> None:
    tr = read_raw(TRAIN_FILE)
    te = read_raw(TEST_FILE)
    feats = feature_columns(tr)
    assert list(tr.columns) == list(te.columns), "esquemas distintos"
    te = te.astype(tr.dtypes.to_dict())  # mismos dtypes => hashes comparables entre archivos

    rep_tr, g_tr = file_report(TRAIN_FILE, tr, feats)
    rep_te, g_te = file_report(TEST_FILE, te, feats)

    splits = pd.read_csv(PREP_DIR / "splits.csv")
    manifest = json.loads((PREP_DIR / "manifest.json").read_text())
    g_train = set(splits.loc[splits.split == "train", "group_id"])
    g_val = set(splits.loc[splits.split == "val", "group_id"])
    te_hex = g_te

    audit = {
        "generated": str(date.today()),
        "download": SOURCES["download"],
        "content_check": SOURCES["content_check"],
        "files": {"training": rep_tr, "testing": rep_te},
        "schema_identical_train_test": True,
        "target": {"column": TARGET, "positive_class": "1 = ataque", "negative_class": "0 = normal"},
        "variables": {
            "used": feats,
            "categorical": CATEGORICAL,
            "numeric": [c for c in feats if c not in CATEGORICAL],
            "excluded": {"id": "identificador de fila (solo auditoría y trazabilidad)",
                         "attack_cat": "revela la respuesta (fuga de etiqueta)",
                         "label": "variable objetivo"},
        },
        "cleaning_policy": {
            "rows_removed": 0,
            "duplicates": "Se conservan: reflejan la frecuencia real de conexiones. Se agrupan por hash de las "
                          "42 variables predictoras y cada grupo va completo a train o a val.",
            "contradictory_groups": "Se conservan en train y val: eliminarlas exigiría mirar la etiqueta y el "
                                    "test también las contiene; representan error irreducible y se reportan.",
            "missing_values": "No hay faltantes; aun así el pipeline incluye imputación ajustada solo con train.",
            "categorical_values": "'-' en service se trata como una categoría más; categorías nuevas en "
                                  "val/test se codifican como desconocidas (handle_unknown='ignore').",
            "test_file": "Reservado. Solo se auditan esquema, duplicados y cruces; sus etiquetas no se usan "
                         "para elegir modelo ni umbral.",
        },
        "splits": {
            "seed": SEED,
            "method": manifest["method"],
            "row_id_rule": manifest["row_id_rule"],
            "train": manifest["splits"]["train"],
            "val": manifest["splits"]["val"],
            "splits_csv": "data/prepared/splits.csv",
            "splits_csv_sha256": manifest["splits_csv_sha256"],
            "groups_shared_train_val": len(g_train & g_val),
        },
        "cross_partition_overlap": {
            "test_rows": int(len(te)),
            "test_rows_with_features_in_train_split": int(te_hex.isin(g_train).sum()),
            "test_rows_with_features_in_val_split": int(te_hex.isin(g_val).sum()),
            "test_rows_with_features_in_train_or_val": int(te_hex.isin(g_train | g_val).sum()),
            "test_unique_groups_in_train_or_val": int(pd.Series(te_hex.unique()).isin(g_train | g_val).sum()),
            "test_unique_groups": int(te_hex.nunique()),
            "note": "El solapamiento existe en las particiones publicadas; las métricas de test incluyen filas cuyas "
                    "entradas exactas ya se vieron en entrenamiento y deben interpretarse con esa limitación.",
        },
    }
    ov = audit["cross_partition_overlap"]
    ov["test_overlap_fraction"] = round(ov["test_rows_with_features_in_train_or_val"] / ov["test_rows"], 4)
    write_json(EVID_DIR / "auditoria.json", audit)
    print(json.dumps({k: audit["files"][k]["duplicates_by_features"] | audit["files"][k]["contradictory_label_groups"]
                      for k in audit["files"]}, indent=1))
    print(json.dumps(ov, indent=1))


if __name__ == "__main__":
    main()
