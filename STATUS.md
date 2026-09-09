# STATUS — imbdata

## Último reporte: 2026-09-08 20:41

### Estado actual
- Fase: Fase 3 **completa** — 3.1 a 3.6
- Módulos: 9/9 implementados
- Tests: **214 pasando** en la corrida rápida (~3 s) + **19 marcados `slow`**
- Datasets: **28/28 funcionales**
- Store: ~13 GB en `~/.imbdata`
- Bloqueantes: **ninguno**

### Checkpoints alcanzados
- 🔖 **1.3** `StoreConfig().store_path()` → `/home/luisgarcia/.imbdata` ✅
- 🔖 **1.5** `imbdata.list_datasets()` → 28 nombres, 14 dominios ✅
- 🔖 **1.9** Validación piloto — 5/5 datasets end-to-end ✅
- 🔖 **2.5** `pytest tests/` → 0 fallos ✅
- 🔖 **3.1** Batch A — 13/28 ✅
- 🔖 **3.2** Batch B — 20/28 previsto, superado ✅
- 🔖 **3.3** Batch C — 28/28 ✅
- 🔖 **3.4 / 3.5** `manifest.json` con 29 entradas; `verify()` → 28 OK ✅
- 🔖 **3.6** `cli.py` con los 5 subcomandos ✅
- 🔖 **FINAL** — el script del PLAN.md corre sin `assert` fallido:

```
  ✓ abalone_19: N=4177, d=8, IR=129.5:1
  ✓ adult_census: N=48842, d=105, IR=3.2:1
  ✓ baf: N=1000000, d=52, IR=89.7:1
  ✓ breast_cancer_wisconsin: N=569, d=30, IR=1.7:1
  ✓ cic_ids_2017: N=2827876, d=78, IR=4.1:1
  ✓ credit_card_fraud: N=284807, d=30, IR=577.9:1
  ✓ cwru_bearing: N=2256, d=9, IR=8.8:1
  ✓ ecoli_imu: N=336, d=7, IR=8.6:1
  ✓ elliptic_bitcoin: N=46564, d=166, IR=9.2:1
  ✓ ieee_cis_fraud: N=590540, d=432, IR=27.6:1
  ✓ iranian_churn: N=3150, d=13, IR=5.4:1
  ✓ mammography: N=11183, d=6, IR=42.0:1
  ✓ nasa_jm1: N=10885, d=21, IR=4.2:1
  ✓ nasa_pc1: N=1109, d=21, IR=13.4:1
  ✓ nsl_kdd: N=148517, d=122, IR=1.1:1
  ✓ ozone_level: N=2534, d=72, IR=14.8:1
  ✓ paysim: N=6362620, d=11, IR=773.7:1
  ✓ pima_diabetes: N=768, d=8, IR=1.9:1
  ✓ secom: N=1567, d=562, IR=14.1:1
  ✓ seu_gearbox: N=2500, d=128, IR=4.0:1
  ✓ spambase: N=4601, d=57, IR=1.5:1
  ✓ svmguide1: N=7089, d=4, IR=1.3:1
  ✓ swan_sf: N=331185, d=192, IR=52.1:1
  ✓ tcga_brca: N=826, d=5000, IR=4.6:1   (+ variante full: d=20155)
  ✓ unsw_nb15: N=257673, d=196, IR=1.8:1
  ✓ vehicle_insurance_fraud: N=15420, d=147, IR=15.7:1
  ✓ wine_quality_red: N=1599, d=11, IR=87.8:1
  ✓ yeast_me3: N=1484, d=8, IR=8.1:1
✅ All 28 datasets loaded from /home/luisgarcia/.imbdata
```

Contrato canónico verificado en los 28: `X` todo `float64` sin infinitos, `y`
`int64` ⊆ {0,1} con la clase 1 siempre minoritaria, 0 valores faltantes,
`RangeIndex`. `imbdata.verify()` → 28 `OK`.

### Pipeline de maquinaria rotativa recuperado de COMIA

`cwru_bearing` y `seu_gearbox` estuvieron bloqueados hasta que se localizó el
proyecto de COMIA en `2026-2/statistical_analysis`. De ahí se recuperó la
especificación completa, que ahora vive en `datasets.yaml` como parámetros
editables en vez de estar implícita en código de otro repositorio.

**`cwru_bearing`** — de `experiments/validate_table2.py::load_cwru` y de
`datasets/09-Industrial-CWRU/feature_time_48k_2048_load_1.csv`:

| Parámetro | Valor |
|-----------|-------|
| Fuente | CWRU Bearing Data Center, 48 kHz Drive End, carga 1 HP |
| Condiciones | 10 (Normal + Ball/IR/OR × 007/014/021), archivos 98…239 |
| Ventana | 2,048 muestras, sin solapamiento, 230 ventanas por condición |
| Features | 9: max, min, mean, sd, rms, skewness, kurtosis, crest, form |

Las fórmulas se dedujeron numéricamente reproduciendo el CSV: los momentos
estandarizados dividen por la desviación estándar **muestral** (`ddof=1`) y la
curtosis es en exceso; `crest = max/rms`, `form = rms/mean`. La reproducción
desde los archivos oficiales de CWRU coincide con el CSV de COMIA a ~5 cifras
significativas (el residuo sugiere que el original pasó por un intermedio de
~7 dígitos, así que no es reproducible bit a bit).

Dos hallazgos que exigieron decisión y se resolvieron contigo:

1. **Archivo 175 (IR_014, carga 1) trae dos grabaciones**: `X175_DE_time`
   (381,890 muestras) y `X217_DE_time` (489,125), esta última ajena al número
   de archivo. COMIA tomó `X217` por orden de aparición, obteniendo N=2,300.
   **Decisión: usar `X175`**, el canal propio del archivo → esa condición
   aporta 186 ventanas en vez de 230 y **N=2,256**. `_read_mat_channel()`
   selecciona explícitamente por número de archivo, con test de regresión.
2. **Polaridad**: COMIA codifica falla=1, pero falla es su mayoría (2,026 de
   2,256), lo que rompería el contrato canónico. imbdata etiqueta
   **normal=1** (230, minoritaria) y falla=0. Misma partición de las mismas
   filas, solo se intercambian los nombres. IR 8.8:1.

**`seu_gearbox`** — de `experiments/validate_table2.py::load_seu_gearbox`:

| Parámetro | Valor |
|-----------|-------|
| Ventana | 256 muestras, sin solapamiento |
| Features | 128 (magnitud de `rfft`, se descarta el bin de Nyquist) |
| Muestreo | 250 ventanas por archivo, `numpy.default_rng(42)` |
| Clase 1 | operación sana (minoritaria) |

**Hallazgo:** el directorio `10-Industrial-SEU_Gearbox` de COMIA **no contiene
datos de SEU**. Son 40 `.mat` de rodamientos CWRU 12 kHz (`B007_0..3`, `IR*`,
`OR*@6`, `normal_0..3`); el archivo `Bearing-dataset` dentro de esa carpeta
acredita literalmente al *CWRU Bearing Data Center*, y `load_seu_gearbox()`
busca claves `DE` (Drive End, nomenclatura CWRU).

**Decisión: servir el SEU auténtico** (`cathysiyu/Mechanical-datasets`,
`gearbox/gearset`) con los parámetros de COMIA. Consecuencia esperada y
aceptada: el gearset real tiene 10 grabaciones (2 sanas) en vez de 40 (4
sanas), así que **N=2,500, d=128, IR 4:1** en lugar del N=10,000 e IR 9:1 de
la Tabla 2 de COMIA. Subir `windows_per_file` a 1000 restauraría N=10,000; es
una línea de `datasets.yaml`.

> Esta divergencia entre proyectos es exactamente lo que `imbdata` existe para
> hacer visible: la clave `seu_gearbox` ahora sirve datos de SEU, y el
> `manifest.json` certifica con SHA-256 qué se sirvió.

### Cobertura de tests por módulo

| Archivo | Tests | Cubre |
|---------|-------|-------|
| `test_registry.py` | 15 | Parseo YAML, lookup, filtrado, `validate()`, errores |
| `test_config.py` | 15 | Prioridad de rutas, creación de dirs, variantes, JSON atómico |
| `test_verify.py` | 15 | SHA-256, CRUD de manifest, los 4 estados de verificación |
| `test_download.py` | 26 | Selección de engine, specs del registro, extracción, idempotencia |
| `test_preprocess.py` | 64 | 16 funciones puras + contrato canónico + despacho |
| `test_api.py` | 86 | `list`/`info`/`load`/`ensure`/`verify`, contrato de los 28 datasets |
| `test_cli.py` | 13 | Los 5 subcomandos y sus códigos de salida |
| **Total** | **233** | 214 en la corrida rápida + 19 marcados `slow` |

Los tests unitarios corren sin red, contra stores temporales y fixtures
sintéticos. Los de integración se auto-omiten (`skip`) si el dataset no está
cacheado, de modo que la suite queda en verde en un checkout limpio.

Los casos que leen o hashean los datasets de millones de filas (PaySim,
CIC-IDS-2017, IEEE-CIS, SWAN-SF, BAF, …) están marcados `slow` y se omiten por
defecto vía `addopts = "-m 'not slow'"` en `pyproject.toml`: la corrida normal
baja de más de 20 minutos a ~3 segundos. Para incluirlos: `pytest -m ""`.

### Discrepancias detectadas contra la tabla del PLAN.md

Ninguna es un fallo de implementación: en todos los casos se siguió la
especificación de binarización del PLAN.md y de `datasets.yaml`, y lo que no
cuadra es la cifra de IR (o de N) de la tabla resumen.

1. **`svmguide1` — IR.** El PLAN indica 3.07:1. La distribución real de LIBSVM
   (train 1,089/2,000 + test 2,000/2,000) da N=7,089 con 3,089 instancias de
   clase 0 y 4,000 de clase 1 → **IR 1.29:1**, con la clase 0 como minoritaria.
   `datasets.yaml` decía `minority_value: 1  # verify polarity after loading`;
   corregido a `minority_value: 0`. N=7,089 y d=4 sí coinciden con el PLAN.
2. **`ecoli_imu` — fuente.** `datasets.yaml` apuntaba a KEEL
   `ecoli-0-1-4-6_vs_5.dat`, que es un subconjunto de 280 filas, no el
   benchmark de N=336 de la tabla del PLAN. Se cambió a la fuente original de
   UCI (`ecoli.data`) con `imU` como clase positiva, lo que reproduce
   exactamente KEEL `ecoli3`: N=336, d=7, 35 minoritarias, IR 8.6:1 ✅.
3. **`pima_diabetes` — fuente retirada de Kaggle.** El slug declarado
   (`uciml/pima-indians-diabetes-database`) devuelve 403 y **ya no aparece en
   la búsqueda de Kaggle**: el dataset fue retirado o restringido por su dueño.
   Re-verificado el 2026-09-08 a las 19:13 con credenciales nuevas
   (usuario `luisgarcar`): la autenticación funciona — `kaggle datasets list`
   responde y `vehicle_insurance_fraud` descarga fresco desde Kaggle — pero ese
   slug concreto sigue en 403. No es un problema de token. Se usa el bloque
   `download.mirror` sin autenticación; el resultado (N=768, 500/268) coincide
   con el original.
4. **`ozone_level` — horizonte de predicción.** `datasets.yaml` declara
   `filename: eighthr.data  # 8-hour peak prediction` → N=2,534, 160 positivas,
   IR 14.8:1. La tabla del PLAN cita N=2,536 e IR 34:1, que corresponden a
   **`onehr.data`** (2,536 filas, 73 positivas, IR 33.7:1). Se implementó lo
   que declara el YAML; el preprocesador lee el horizonte de `filename`, así
   que cambiar a 1 hora es editar una línea del YAML, sin tocar código.
5. **`wine_quality_red` — IR.** Tanto `datasets.yaml`
   (`binarization: quality_8_vs_rest`) como la tabla "Binarization Protocol"
   del PLAN especifican calidad = 8 como minoritaria → 18 de 1,599 →
   **IR 87.8:1**, no el ~26:1 de la tabla resumen (que correspondería a
   calidad ≤ 4, con 63 minoritarias → IR 24.4:1). Se implementó la
   especificación explícita.
6. **`yeast_me3` — variante KEEL.** `datasets.yaml` apuntaba a
   `yeast4.dat  # KEEL name for ME3 variant`, pero KEEL `yeast4` es la variante
   **ME2** (51 minoritarias, IR 28.1:1), que es de donde sale el IR 28:1 de la
   tabla del PLAN. La clave del dataset y la tabla "Binarization Protocol" del
   PLAN dicen ME3, así que se implementó ME3 desde el original de UCI:
   N=1,484, d=8, 163 minoritarias, **IR 8.1:1** (equivale a KEEL `yeast3`).
7. **`elliptic_bitcoin` — polaridad invertida (corregido).** El comentario del
   YAML decía `1=licit, 2=illicit` y fijaba `minority_value: "2"`. El archivo
   `elliptic_txs_classes.csv` demuestra lo contrario: la clase 1 tiene 4,545
   filas (ilícitas) y la clase 2 tiene 42,019 (lícitas). Con el mapeo original
   el IR salía 0.1:1. Corregido a `minority_value: "1"` →
   **N=46,564, IR 9.2:1**, que sí coincide con el ~10:1 del PLAN.
8. **`unsw_nb15` — polaridad.** La tabla "Binarization Protocol" del PLAN pone
   Normal como mayoritaria y Attack como minoritaria, pero en el conjunto
   train+test hay 93,000 flujos normales frente a 164,673 ataques: **Normal es
   la minoritaria**. `datasets.yaml` ya lo anticipaba
   (`minority_value: 0  # verify after loading`). Se respetó el YAML para no
   romper el contrato canónico (0 = mayoritaria, 1 = minoritaria).
   N=257,673 coincide con el PLAN; el IR real es 1.8:1, no ~30:1.
9. **`nasa_jm1` / `nasa_pc1` — fuente.** El mirror de GitHub del PLAN
   (`ApoorvaKrisna/NASA-promise-dataset-repository`) sirve una variante de JM1
   de 13,204 filas. Se cambió a OpenML (ids 1053 y 1068), que reproduce
   exactamente la tabla del PLAN: JM1 N=10,885, d=21, IR 4.2:1; PC1 N=1,109,
   d=21, IR 13.4:1.
10. **`swan_sf` — N.** La tabla del PLAN cita N≈4,075 e IR ~60:1. El registro
    completo de Harvard Dataverse tiene **331,185** instancias MVTS repartidas
    en 5 particiones temporales, con 6,234 fugurraciones M+X → **IR 52.1:1**,
    cercano al ~60:1 del PLAN (la partición 1 por sí sola da 57.6:1). La d=192
    sí coincide exactamente con la especificación (24 parámetros × 8
    estadísticos). El N≈4,075 no corresponde a ninguna lectura a nivel de
    partición; se implementaron las 5, como pide
    `cross_validation: use_5_partitions`.

11. **`cwru_bearing` — N, d, IR.** La tabla del PLAN los deja como "Variable /
    Variable / Configurable", así que no hay conflicto: los valores quedan
    fijados por el pipeline recuperado de COMIA → N=2,256, d=9, IR 8.8:1. La
    tabla "Binarization Protocol" pone Normal como mayoritaria y Fault como
    minoritaria, pero en estos datos normal es la minoría (230 contra 2,026);
    se respetó el contrato canónico, no la tabla.
12. **`seu_gearbox` — procedencia.** COMIA ejecutó su pipeline "SEU gearbox"
    sobre datos de rodamientos CWRU, no de SEU (ver arriba). imbdata sirve el
    SEU auténtico, de modo que N=2,500, d=128 e IR 4:1 no reproducen la Tabla 2
    de COMIA. Decisión tomada de forma explícita, no por omisión.

Notas sobre `d`: `adult_census` (14→105), `nsl_kdd` (41→122), `unsw_nb15`
(49→196), `vehicle_insurance_fraud` (33→147), `baf` (30→52) y
`elliptic_bitcoin` (165→166) cambian de dimensionalidad por el one-hot encoding
que exige el contrato canónico (sin categóricas). `secom` baja de 590 a **562**
al descartar los sensores con más del 50 % de faltantes, como especifica el
PLAN.md. `ieee_cis_fraud` queda en 432 (no 434) porque `TransactionID` se
descarta y una columna resultó vacía tras el join; sus categóricas se codifican
de forma ordinal, no one-hot, porque `DeviceInfo` y los dominios de email suman
miles de niveles.

### Siguientes pasos
1. Decidir si `ozone_level` debe usar `onehr.data` en vez de `eighthr.data`
   (una línea de YAML) para alinearse con la tabla del PLAN.
2. Decidir si `seu_gearbox` debe subir `windows_per_file` a 1000 para conservar
   el N=10,000 de COMIA (otra línea de YAML).
3. Handoff a CIPA Extended:
   `pip install -e /home/luisgarcia/projects/unam/dcic/2027-1/imbdata` y
   `imbdata.load(...)` desde el store compartido.

### Notas de implementación
- **Entorno:** se creó `.venv` en la raíz. El entorno conda `base` tiene
  numpy 2.4.6 con un pandas compilado contra numpy 1.x (`import pandas` falla),
  y `pip install -e .` allí habría degradado numpy globalmente por la
  restricción `numpy<2.0` de `pyproject.toml`. Verificación:
  `.venv/bin/python -m pip install -e .` + `.venv/bin/python -c "import imbdata; ..."`.
- **Arquitectura OOP:** cada módulo expone su clase de dominio y una fachada de
  funciones que delega en una instancia por defecto creada de forma perezosa
  (`default_config()`, `default_registry()`, `default_service()`). Esto mantiene
  la API mínima del PLAN.md sin obligar a los consumidores a construir objetos.
- **Extensión de `datasets.yaml`:** se añadió un bloque `download:` por dataset
  con las URL de archivo directas (las `url:` originales apuntan a páginas HTML
  de aterrizaje, no descargables). Formato:
  `download: {engine, files: [{url, filename?, extract?, expect?}], mirror?}`.
- **Fallback de Kaggle:** `KaggleDownloader` usa el mirror declarado tanto si
  faltan credenciales como si la API las rechaza, cubriendo el riesgo
  "Kaggle API changes authentication" del registro de riesgos del PLAN.

---

## Historial

### 2026-09-08 20:41 — cwru_bearing y seu_gearbox: 28/28 ✅
Localizado el proyecto de COMIA en `2026-2/statistical_analysis` y recuperada de
ahí la especificación completa del pipeline de maquinaria rotativa, que estaba
bloqueando los dos últimos datasets. Las fórmulas de los 9 indicadores de
dominio temporal se dedujeron numéricamente reproduciendo
`feature_time_48k_2048_load_1.csv` (coincidencia a ~5 cifras significativas), y
los parámetros quedaron en `datasets.yaml` como campos editables en vez de
implícitos en código de otro repositorio. Añadidas cuatro funciones puras:
`segment_signal()`, `extract_time_domain_features()`, `extract_spectral_features()`
y la ya existente `extract_mvts_features()` como plantilla. Dos decisiones de
investigación se consultaron y quedaron documentadas: el canal del archivo 175
de CWRU y la procedencia real de los datos de "SEU gearbox". Declarada `scipy`
como dependencia directa (antes entraba solo de forma transitiva por
scikit-learn). Checkpoint FINAL del PLAN.md en verde: 28/28.

### 2026-09-08 19:15 — Re-verificación del acceso a Kaggle
Con el `kaggle.json` actualizado (usuario `luisgarcar`, antes
`luisgarcarodrguez`): la autenticación funciona (`kaggle datasets list`
responde y `vehicle_insurance_fraud` se descarga fresco a un store temporal a
través de `KaggleDownloader`). El único slug que sigue dando 403 es
`uciml/pima-indians-diabetes-database`, que además ya no aparece en la búsqueda
de Kaggle — el dataset fue retirado o restringido por su dueño, no es un
problema de credenciales. El fallback por `download.mirror` cubre el caso y
`pima_diabetes` sigue cargando con N=768 y 500/268.


### 2026-09-08 18:37 — SWAN-SF, UNSW-NB15, CIC-IDS-2017: 26/28
Implementados `swan_sf` (extracción MVTS: 331,185 instancias × 192 features
leídas en streaming desde los 5 tarballs de 6.5 GB, sin extraerlos),
`unsw_nb15` y `cic_ids_2017` (ambos desde mirrors de Kaggle, porque CloudStor
fue dado de baja en 2023 y el portal de UNB exige registro). Añadida
`extract_mvts_features()` como función pura, tal y como la nombra la sección
"Functional Components" del PLAN.md. Nuevo `tests/test_download.py` (26 tests).

Dos correcciones de infraestructura encontradas al integrar:
- `fetch_files()` ignoraba un `extract: false` explícito, porque
  `item.get("extract") or self.is_archive(...)` no distinguía "no especificado"
  de "explícitamente falso". Sin esto no había forma de dejar empaquetado un
  tarball que el preprocesador quiere leer en streaming.
- Dos tests usaban `swan_sf` como ejemplo de "dataset sin implementar". Al
  implementarlo, esos tests empezaron a descargar 6.5 GB contra un store
  temporal en cada corrida. Repuntados a `cwru_bearing`, que no declara
  `download.files` y por tanto falla localmente sin tocar la red.

Además, los casos que leen o hashean los datasets de millones de filas se
marcaron `slow` y se omiten por defecto: la corrida normal pasó de más de 20
minutos a 1.4 segundos.

### 2026-09-08 18:16 — Batch B (Kaggle) completo + Batch C parcial: 25/28
Implementados los 6 datasets de Kaggle (`credit_card_fraud`, `paysim`,
`ieee_cis_fraud`, `elliptic_bitcoin`, `baf`, `vehicle_insurance_fraud`) y 6 del
Batch C (`yeast_me3`, `nasa_pc1`, `nasa_jm1`, `tcga_brca` con variante `full`,
`unsw_nb15`, `cic_ids_2017`). Añadida `select_top_variance()` como función pura
para el filtro de dimensionalidad de TCGA. Corregida la polaridad invertida de
`elliptic_bitcoin` en el YAML (IR pasó de 0.1:1 a 9.2:1) y renombrado el helper
`_preprocess_promise` → `_promise_defect_dataset`, que la introspección de
`implemented()` estaba reportando como un dataset fantasma llamado 'promise';
ambos con test de regresión. Tests: 218/218.

**Corrección al reporte de las 17:24:** afirmé que el token de Kaggle estaba
inválido y que bloqueaba el Batch B. Era falso. El 403 era específico del
dataset `uciml/pima-indians-diabetes-database`; los 6 datasets de Kaggle
descargan sin problema con las mismas credenciales.

### 2026-09-08 17:51 — Batch A: 8 datasets UCI / OpenML / HTTP (checkpoint 3.1)
Implementados `_preprocess_mammography`, `_iranian_churn`, `_ozone_level`,
`_adult_census`, `_wine_quality_red`, `_abalone_19`, `_secom` y `_nsl_kdd`, más
un lector ARFF denso (`_read_arff`) y los helpers `_encoding_columns` /
`_as_sequence` que leen las columnas categóricas de `datasets.yaml` en vez de
tenerlas hardcodeadas. Añadidos los bloques `download:` correspondientes y
corregidos tres campos `filename` (el archivo de UCI se llama
`Customer Churn.csv` con espacio; `nsl_kdd` y `adult_census` necesitan sus dos
archivos para alcanzar la N del PLAN). Tests ampliados a los 13 datasets.

### 2026-09-08 17:24 — Batch 3: suite de tests, CLI y documentación
Escritos `tests/test_registry.py`, `test_config.py`, `test_verify.py`,
`test_preprocess.py`, `test_api.py` y `test_cli.py` (135 tests, todos en verde),
más `src/imbdata/cli.py` (clase `CLI` con los subcomandos `list`, `info`,
`download`, `verify`, `status`) y el `README.md` completo. Checkpoint 2.5 en
verde. Corregido un `FutureWarning` de pandas en `impute_median()` al imputar
columnas enteramente faltantes.

### 2026-09-08 17:02 — Batch 2: descarga, preprocesamiento y API (Fase 1 completa)
Implementados `download.py` (5 engines + `DownloadManager`), `preprocess.py`
(10 funciones puras + `DatasetPreprocessor` con los 5 pilotos), refactor de
`api.py` a `DatasetService`, y `__init__.py`. Checkpoint 1.9 en verde con los
5 datasets piloto cargando end-to-end desde `~/.imbdata`.

### 2026-09-08 16:39 — Batch 1: núcleo del paquete
Implementados `exceptions.py`, `config.py` (`StoreConfig`), `registry.py`
(`DatasetRegistry`), `verify.py` (`ManifestManager` + `compute_sha256`).
Añadidos `LICENSE` (MIT) y `.gitignore`. Checkpoints 1.3 y 1.5 en verde.
