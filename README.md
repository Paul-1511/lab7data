# lab7data

Laboratorio 7 de CC3066 Data Science (segundo semestre de 2026): análisis de las bases de Personas de la ENEIC con Python, Spark 3.5 y `pyspark.ml`. El objetivo es perfilar a los trabajadores asalariados con KMeans y estimar el salario mensual con regresión lineal y Random Forest bajo una validación temporal estricta.

## Autoría y avance previo

Las contribuciones de cada integrante se registran en el historial Git del repositorio. `main7.py` y `lab7.ipynb` son el avance previo del compañero y se conservan sin cambios. Sus cifras preliminares no se usan como resultados finales porque se calcularon con 4,000 filas por período y con una partición aleatoria. La solución final está en `lab7_final.py` y `lab7_final.ipynb`.

## Arquitectura

| Archivo o carpeta | Propósito |
|---|---|
| `lab7_final.py` | CLI modular, reanudable y validable por fases |
| `lab7_final.ipynb` | Notebook de entrega: carga solo resultados persistidos, sin reentrenar |
| `Dockerfile.lab7`, `compose.lab7.yml`, `.dockerignore` | Entorno Linux reproducible y aislado |
| `requirements.txt` | Versiones fijadas de Python |
| `outputs/foundation/` | Manifiesto, auditorías de faltantes, exclusiones, llaves y códigos |
| `outputs/eda/` | Descriptivos, categorías, correlación, KMeans, perfiles y figuras |
| `outputs/models/` | Selección temporal: configuraciones, métricas y figura de RMSE |
| `outputs/test_2026/` | Prueba final: métricas, errores por grupo y banda, conclusiones y figuras |
| `outputs/delivery/` | Validación de la entrega |
| `data_processed/` | Parquet de elegibles, clusters y predicciones (ignorado por Git) |
| `models/` | Pipelines Spark persistidos (ignorado por Git) |

Fases de `lab7_final.py`:

1. `--preflight`: verifica versiones, Java, `ps`, fuentes, diccionarios y una prueba de escritura y lectura Parquet.
2. `--foundation` / `--validate-foundation`: lee un libro a la vez y solo 14 columnas, con tipos explícitos, filtros secuenciales auditados y Parquet separados de 2025 y 2026.
3. `--eda-clustering` / `--validate-eda-clustering`: estadísticas con Spark, correlación de Pearson y KMeans con K=2..5.
4. `--models-validation` / `--validate-models-validation`: baseline, regresión lineal y Random Forest, con entrenamiento en 2025T1-T3 y selección en 2025T4.
5. `--final-test` / `--validate-final-test`: reentrena las configuraciones congeladas con todo 2025 y las evalúa una sola vez en 2026T1.
6. `--validate-delivery`: revisa fases, notebook, README, Docker y repositorio sin recalcular.

Cada validación lee solo artefactos persistidos y termina con `=OK` o `=FALLA`.

## Datos

Los cinco libros oficiales de Personas de la ENEIC van en `datos/` y sus diccionarios en `datos/diccionarios/`:

| Archivo | Hoja | Filas | Columnas | Elegibles |
|---|---|---:|---:|---:|
| `Personas_2025T1.xlsx` | `Personas_ENEIC_T1_2025` | 51,588 | 270 | 13,419 |
| `Personas_2025T2.xlsx` | `Personas ENEIC T2-2025` | 51,167 | 270 | 13,492 |
| `Personas_2025T3.xlsx` | `Personas ENEIC T3-2025` | 51,583 | 270 | 13,450 |
| `Personas_2025T4.xlsx` | `Base de datos Personas ENEIC IV` | 49,338 | 302 | 12,664 |
| `Personas_2026T1.xlsx` | `Personas_ENEIC_T1-2026` | 49,843 | 270 | 13,258 |

**Lectura y trazabilidad**

- **Selección por nombre:** las columnas se eligen por nombre y los períodos se unen con `unionByName`, porque 2025T4 tiene otro esquema.
- **Período:** `periodo_archivo` se deriva del nombre del archivo; el `TRIMESTRE` original se conserva sin ajustes.
- **Llave:** `(periodo_archivo, NUM_HOGAR, NUM_PERSONA)` es única en las 253,519 filas.

**Filtros de la población analítica, en orden**

1. Edad ≥ 15.
2. `OCUPADOS == 1` y `P05C16` en {1, 2, 3, 4}.
3. Salario > 0.
4. `P05C07A` ≥ 0 y `P05C07B` entero entre 0 y 11.
5. Antigüedad ≤ edad.
6. Horas semanales en (0, 168].

**Particiones:**

| Partición | Períodos | Filas |
|---|---|---:|
| Desarrollo | 2025T1-T3 | 40,361 |
| Validación temporal | 2025T4 | 12,664 |
| Reentrenamiento final | Todo 2025 | 53,025 |
| Prueba final | 2026T1 | 13,258 |

**Variables del modelo:**

- Objetivo: `salario_mensual = P05D01`, sin imputar, recortar ni transformar.
- Predictores numéricos: edad, antigüedad y horas semanales.
- Predictores categóricos: nivel educativo, categoría ocupacional y dominio.

## Entorno Docker

`Dockerfile.lab7` parte de la imagen pública `python:3.11.16-slim-bookworm`. Instala Java 17 (`openjdk-17-jre-headless`), `procps` y las dependencias fijadas en `requirements.txt`, entre ellas PySpark 3.5.1, pandas 3.0.6, pyarrow 25.0.1, numpy 2.4.6, openpyxl 3.1.5, matplotlib 3.11.2 y seaborn 0.13.2. No depende de imágenes locales previas.

`compose.lab7.yml` construye la imagen `lab7-pyspark:latest` y usa el contenedor `lab7-pyspark`. Monta la raíz del repositorio como directorio de trabajo y publica Jupyter solo en `127.0.0.1:8888`. `.dockerignore` limita el contexto de construcción a `Dockerfile.lab7` y `requirements.txt`.

Se recomienda asignar al menos 4 GB de memoria a Docker Desktop. Spark se ejecuta en `local[2]` con 1 GB para el driver.

## Comandos reproducibles

Desde PowerShell, en la raíz del repositorio:

```powershell
docker compose -f compose.lab7.yml build
docker compose -f compose.lab7.yml run --rm lab7 python lab7_final.py --preflight
docker compose -f compose.lab7.yml run --rm lab7 python lab7_final.py --foundation
docker compose -f compose.lab7.yml run --rm lab7 python lab7_final.py --validate-foundation
docker compose -f compose.lab7.yml run --rm lab7 python lab7_final.py --eda-clustering
docker compose -f compose.lab7.yml run --rm lab7 python lab7_final.py --validate-eda-clustering
docker compose -f compose.lab7.yml run --rm lab7 python lab7_final.py --models-validation
docker compose -f compose.lab7.yml run --rm lab7 python lab7_final.py --validate-models-validation
docker compose -f compose.lab7.yml run --rm lab7 python lab7_final.py --final-test
docker compose -f compose.lab7.yml run --rm lab7 python lab7_final.py --validate-final-test
docker compose -f compose.lab7.yml run --rm lab7 jupyter nbconvert --to notebook --execute --inplace lab7_final.ipynb
docker compose -f compose.lab7.yml run --rm lab7 python lab7_final.py --validate-delivery
```

Jupyter interactivo: `docker compose -f compose.lab7.yml up`. Después se abre el enlace con token que aparece en la consola.

`--final-test` evalúa 2026T1 una sola vez; si la prueba ya existe no la repite, salvo que se use `--force` de forma explícita.

## Resultados principales

**Fundación:** 53,025 elegibles en 2025 y 13,258 en 2026T1, iguales a los conteos de control. Las cuatro fases quedaron validadas sin fallas.

**Descripción del salario en 2025**

- Media Q3,421.68, mediana Q3,000; P25 Q1,800, P75 Q4,000 y P95 Q8,000.
- Muy asimétrico (asimetría 6.00). Los registros ≥ P95 concentran el 19.0 % de la masa salarial.
- La mediana crece con la educación: de Q1,500 (Ninguno) a Q10,000 (Maestría).

**Correlación de Pearson:** débil entre el salario y edad (0.147), antigüedad (0.180) y horas (0.076); moderada entre edad y antigüedad (0.487).

**KMeans (sin salario en el ajuste)**

- K=2 obtuvo la mejor silueta entre K=2..5 (0.4666); es el mejor entre los valores probados, no un óptimo universal.
- Separa trabajadores jóvenes con poca antigüedad (70.1 %) de trabajadores mayores con trayectorias largas (29.9 %).

**Selección en 2025T4 (entrenamiento con 40,361 filas y validación con 12,664)**

| Modelo | RMSE | MAE | R² |
|---|---:|---:|---:|
| Baseline (media de T1-T3) | 2,889.57 | 1,672.50 | −0.0031 |
| Regresión lineal `lr_reg_0.0` | 2,186.42 | 1,210.14 | 0.4257 |
| Random Forest `rf_t100_d10` | 1,985.60 | 1,074.32 | 0.5263 |

**Prueba única en 2026T1 (13,258 filas, modelos reentrenados con 53,025)**

| Modelo | RMSE | MAE | R² | Error medio |
|---|---:|---:|---:|---:|
| Baseline (media de 2025) | 2,871.42 | 1,718.09 | −0.0025 | 144.11 |
| Regresión lineal | 2,163.75 | 1,240.71 | 0.4307 | 120.55 |
| Random Forest | 1,961.02 | 1,103.68 | 0.5324 | 108.17 |

**Errores en salarios altos:** ambos modelos subestiman sistemáticamente los salarios ≥ P95 (Q8,000; n = 812).

- Regresión lineal: error medio Q4,887.11 y 94.0 % de residuos positivos.
- Random Forest: error medio Q4,252.24 y 91.5 % de residuos positivos.

Además, ambos sobreestiman los salarios bajos (< P25).

## Limitaciones

- **Sin ponderar:** los análisis no usan `FACTOR`. Describen los registros elegibles analizados y no constituyen estimaciones poblacionales del país.
- **Asociaciones, no causas:** los resultados no establecen causalidad ni son recomendaciones salariales.
- **Seis predictores:** con estas variables, los modelos comprimen las predicciones hacia la media. Faltan variables como ocupación, rama de actividad o tamaño de empresa.
- **Pocos casos en algunos grupos:** Doctorado y Maestría tienen muy pocos registros.
- **Rangos acotados:** K y los hiperparámetros son los mejores dentro de los rangos probados. Random Forest ganó con la mayor profundidad ensayada.

## Archivos ignorados

`.gitignore` excluye:

- Datos: `datos/` y `datos/diccionarios/`. Los cinco XLSX de `datos/` se agregaron en commits anteriores y siguen versionados; la regla evita que se agreguen archivos nuevos de esa carpeta, como los diccionarios.
- Datos procesados: `data_processed/` y cualquier `*.parquet` o `*.csv`.
- Modelos binarios de la raíz: `/models/`. Solo la carpeta de la raíz; `outputs/models/` sí se versiona.
- Entornos virtuales, cachés, `.ipynb_checkpoints/`, logs y configuración del IDE.
- Documentos privados de trabajo: `CONTEXTO_CLAUDE_LAB7.md` y `GUIA_REPLICACION_LAB7.md`.

Se versionan el código, el notebook ejecutado, este README, los archivos Docker, `requirements.txt` y los artefactos pequeños de `outputs/` (JSON y PNG) que respaldan los resultados.
