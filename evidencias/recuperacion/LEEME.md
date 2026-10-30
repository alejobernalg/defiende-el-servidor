# Prueba de recuperación

- `entrada_validacion.csv`: 25 filas fijas de validación (`val.sample(n=25, random_state=42)`; 10 normales, 15 ataques).
- `salida_v1_original.csv`, `salida_v2_original.csv`: salidas calculadas con los artefactos originales ANTES de publicar.
- `entorno_nuevo/`: ejecución de `scripts/recuperacion.sh` en una carpeta nueva (clon de GitHub, venv nuevo,
  caché DVC vacía, `HF_HOME` vacío y sin token):
  - `recuperacion.log` / `salida_completa_con_tiempo.log`: comandos reales y su salida (incluye `real 744.66` s).
  - `salida_v1_recuperada.csv`, `salida_v2_recuperada.csv`: salidas con los modelos descargados por SHA.
  - `comparacion_v1.json`, `comparacion_v2.json`: SHA-256 de cada archivo descargado, igualdad de `y_pred` y
    diferencia máxima de `score_attack` (v1: 0.0; v2: 2.2e-16; tolerancia declarada 1e-9).

La tolerancia existe porque el RandomForest (n_jobs=-1) suma las probabilidades de los árboles en paralelo y
el orden de suma en coma flotante puede variar en el último bit.
