"""Añade el resultado del test final a la model card de v2 en una NUEVA revisión de Hugging Face.

Solo se sube README.md: se verifica que model.joblib en la nueva revisión tiene el mismo SHA-256 que la
revisión evaluada (el modelo congelado no cambia). Registra la revisión en evidencias/seleccion.json.
"""
from __future__ import annotations

import json

from huggingface_hub import HfApi

from common import EVID_DIR, ROOT, write_json


def main() -> None:
    sel = json.loads((EVID_DIR / "seleccion.json").read_text())
    t = json.loads((EVID_DIR / "metricas-test.json").read_text())
    hf = sel["huggingface"]["v2"]
    api = HfApi()
    card = (ROOT / "hf_build/v2/README.md").read_text(encoding="utf-8")
    m, cm = t["metrics"], t["confusion_matrix"]["matrix"]
    vm = t["validation_metrics_same_threshold"]
    card += f"""
## Evaluación final en test (una sola vez, solución congelada)
Evaluada la revisión `{t['huggingface_revision_sha']}` (SHA-256 de `model.joblib` `{t['model_file_sha256']}`) sobre
las {t['rows_evaluated']} filas de `UNSW_NB15_testing-set.csv` (SHA-256 `{t['dataset']['sha256']}`), umbral {t['threshold']}.
Esta revisión solo actualiza la documentación: `model.joblib` es idéntico.

| Conjunto | Recall ataque | Precision ataque | FPR | Macro-F1 |
|---|---|---|---|---|
| Validación | {vm['recall_attack']:.4f} | {vm['precision_attack']:.4f} | {vm['fpr']:.4f} | {vm['macro_f1']:.4f} |
| Test | {m['recall_attack']:.4f} | {m['precision_attack']:.4f} | {m['fpr']:.4f} | {m['macro_f1']:.4f} |

Matriz de confusión test: `[[TN={cm[0][0]}, FP={cm[0][1]}], [FN={cm[1][0]}, TP={cm[1][1]}]]`.

**La FPR de test (≈0.17) es mucho mayor que en validación (≈0.05).** El test publicado tiene más tráfico
normal (45 % frente a 32 %) y su tráfico normal sin servicio identificado (`service='-'`) recibe una FPR de
~0.20 frente a ~0.06 en validación: cambio de distribución entre las particiones publicadas. El resultado se
conserva sin reajustar el umbral. Detalle en `evidencias/analisis-errores.json` del repositorio de código.
"""
    ci = api.upload_file(path_or_fileobj=card.encode("utf-8"), path_in_repo="README.md", repo_id=hf["repo"],
                         commit_message="v2: añade resultado del test final a la model card (modelo sin cambios)")
    before = api.get_paths_info(hf["repo"], ["model.joblib"], revision=hf["revision_sha"])[0].lfs.sha256
    after = api.get_paths_info(hf["repo"], ["model.joblib"], revision=ci.oid)[0].lfs.sha256
    assert before == after == t["model_file_sha256"], (before, after)
    sel["huggingface"]["v2_card_with_test"] = {
        "revision_sha": ci.oid, "commit_url": ci.commit_url, "evaluated_revision_sha": hf["revision_sha"],
        "model_file_sha256_unchanged": after}
    write_json(EVID_DIR / "seleccion.json", sel)
    print(ci.oid, "model.joblib sin cambios:", after)


if __name__ == "__main__":
    main()
