"""Recalcula las métricas de test desde evidencias/predicciones-test.csv (sin modelo ni librerías de ML).

Comprueba también que y_true coincide con el archivo de test versionado si está disponible (dvc pull).
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ref = json.loads((ROOT / "evidencias/metricas-test.json").read_text())
thr = ref["threshold"]
tp = fp = tn = fn = bad_rule = 0
with open(ROOT / "evidencias/predicciones-test.csv") as f:
    rows = list(csv.DictReader(f))
for r in rows:
    y, s, p = int(r["y_true"]), float(r["score_attack"]), int(r["y_pred"])
    bad_rule += p != int(s >= thr)
    tp += p == 1 and y == 1
    fp += p == 1 and y == 0
    tn += p == 0 and y == 0
    fn += p == 0 and y == 1
div = lambda a, b: a / b if b else 0.0
rec, prec, fpr = div(tp, tp + fn), div(tp, tp + fp), div(fp, fp + tn)
f1p = div(2 * prec * rec, prec + rec)
rn, pn = div(tn, tn + fp), div(tn, tn + fn)
macro = (f1p + div(2 * pn * rn, pn + rn)) / 2
calc = {"rows": len(rows), "tp": tp, "fp": fp, "tn": tn, "fn": fn, "recall_attack": rec, "precision_attack": prec,
        "fpr": fpr, "macro_f1": macro, "rows_violating_threshold_rule": bad_rule}
print(json.dumps(calc, indent=1))
cm = ref["confusion_matrix"]["matrix"]
ok = [[tn, fp], [fn, tp]] == cm and all(abs(calc[k] - ref["metrics"][k]) < 1e-12 for k in
                                        ("recall_attack", "precision_attack", "fpr", "macro_f1")) and bad_rule == 0
raw = ROOT / "data/raw/UNSW_NB15_testing-set.csv"
if raw.exists():
    with open(raw, encoding="utf-8-sig") as f:
        labels = [int(r["label"]) for r in csv.DictReader(f)]
    same = labels == [int(r["y_true"]) for r in rows]
    print("y_true coincide con data/raw/UNSW_NB15_testing-set.csv:", same)
    ok = ok and same
print("COINCIDE con metricas-test.json" if ok else "NO COINCIDE")
raise SystemExit(0 if ok else 1)
