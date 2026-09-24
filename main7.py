"""Laboratorio 7 - ejercicios 1 a 5 (Spark MLlib 3.5).

Ejecute, por ejemplo:
    python main7.py --data-dir ./datos --output-dir ./salida_lab7

Los nombres de los archivos deben contener 2025T1, 2025T2, 2025T3, 2025T4 o
2026T1.  Se admiten .xlsx, .xls, .csv y directorios Parquet.  El programa no
usa scikit-learn para entrenar modelos.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import pandas as pd
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import StringType, StructField, StructType
from pyspark.ml import Pipeline
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator, RegressionEvaluator
from pyspark.ml.feature import OneHotEncoder, StandardScaler, StringIndexer, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.stat import Correlation

REQUIRED = ["P05D01", "P02A03", "P05C07A", "P05C07B", "P05H01A", "P03A03A", "P05C16", "DOMINIO", "OCUPADOS", "NUM_HOGAR", "NUM_PERSONA", "FACTOR", "ANIO", "TRIMESTRE"]
NUMERIC = ["salario_mensual", "edad", "antiguedad", "horas_semanales"]
CATEGORICAL = ["nivel_educativo", "categoria_ocupacional", "dominio"]
PERIODS = {"2025T1": (2025, 1), "2025T2": (2025, 2), "2025T3": (2025, 3), "2025T4": (2025, 4), "2026T1": (2026, 1)}

def period_from_name(path: Path) -> str:
    text = re.sub(r"[^A-Z0-9]", "", path.name.upper())
    for period in PERIODS:
        if period in text:
            return period
    # formatos frecuentes: Personas I 2025 / Personas 1er trimestre 2025
    year = re.search(r"20(25|26)", text)
    quarter = re.search(r"(?:T|TRIMESTRE)?([1-4])", text)
    if year and quarter:
        candidate = f"20{year.group(1)}T{quarter.group(1)}"
        if candidate in PERIODS:
            return candidate
    raise ValueError(f"No se pudo identificar el período de {path.name}. Renómbrelo, por ejemplo, Personas_2025T1.xlsx")

def discover_sources(data_dir: str) -> dict[str, Path]:
    files = [p for p in Path(data_dir).rglob("*") if p.suffix.lower() in {".xlsx", ".xls", ".csv"} or p.name.endswith(".parquet")]
    found = {}
    for path in files:
        try:
            period = period_from_name(path)
        except ValueError:
            continue
        if period in found:
            raise ValueError(f"Hay más de un archivo para {period}: {found[period]} y {path}")
        found[period] = path
    needed = set(PERIODS)
    if missing := needed - set(found):
        raise FileNotFoundError("Faltan archivos de Personas: " + ", ".join(sorted(missing)))
    return found

def read_source(spark: SparkSession, path: Path, period: str):
    """Lee un archivo individualmente y crea un DataFrame Spark con esquema explícito de texto."""
    if path.name.endswith(".parquet"):
        raw = spark.read.parquet(str(path))
        raw = raw.select(*[F.col(c).cast("string").alias(str(c).strip().upper()) for c in raw.columns])
    else:
        pdf = pd.read_csv(path, dtype=str) if path.suffix.lower() == ".csv" else pd.read_excel(path, dtype=str)
        pdf.columns = [str(c).strip().upper() for c in pdf.columns]
        schema = StructType([StructField(c, StringType(), True) for c in pdf.columns])
        raw = spark.createDataFrame(pdf.where(pd.notna(pdf), None), schema=schema)
    cols = {c.upper(): c for c in raw.columns}
    missing = set(REQUIRED) - set(cols)
    if missing:
        raise ValueError(f"{path.name} no contiene columnas requeridas: {sorted(missing)}")
    base = raw.select(*[F.col(cols[c]).alias(c) for c in REQUIRED])
    year, quarter = PERIODS[period]
    return base.withColumn("archivo_origen", F.lit(path.name)).withColumn("periodo_archivo", F.lit(period)).withColumn("anio_archivo", F.lit(year)).withColumn("trimestre_calendario", F.lit(quarter))

def clean_code(col):
    # Excel suele convertir códigos enteros como 1 a "1.0"; ambos deben ser la misma categoría.
    normalized = F.regexp_replace(F.trim(col), r"\\.0$", "")
    return F.when(col.isNull() | (F.trim(col) == ""), F.lit("DESCONOCIDO")).otherwise(normalized)

def harmonize(df):
    # Decimal con coma, espacios y códigos que llegan como 1.0 se homogeneizan antes de unir/modelar.
    def number(name):
        s = F.regexp_replace(F.trim(F.col(name)), ",", ".")
        return F.when(s.rlike(r"^[+-]?(\\d+(\\.\\d*)?|\\.\\d+)$"), s.cast("double"))
    out = df
    for name in ["P05D01", "P02A03", "P05C07A", "P05C07B", "P05H01A", "OCUPADOS"]:
        out = out.withColumn(name + "_n", number(name))
    return (out
        .withColumn("salario_mensual", F.col("P05D01_n"))
        .withColumn("edad", F.col("P02A03_n"))
        .withColumn("antiguedad_anios", F.col("P05C07A_n"))
        .withColumn("antiguedad_meses", F.col("P05C07B_n"))
        .withColumn("horas_semanales", F.col("P05H01A_n"))
        .withColumn("antiguedad", F.col("antiguedad_anios") + F.col("antiguedad_meses") / F.lit(12.0))
        .withColumn("nivel_educativo", clean_code(F.col("P03A03A"))) # el código 0 se conserva como categoría válida
        .withColumn("categoria_ocupacional", clean_code(F.col("P05C16")))
        .withColumn("dominio", clean_code(F.col("DOMINIO"))))

def missing_table(df):
    selected = REQUIRED + ["archivo_origen", "periodo_archivo"]
    n = df.count()
    row = df.select(*[F.sum(F.when(F.col(c).isNull() | (F.trim(F.col(c)) == ""), 1).otherwise(0)).alias(c) for c in selected]).first().asDict()
    return [{"variable": c, "faltantes": int(row[c] or 0), "porcentaje": round(100 * (row[c] or 0) / n, 3) if n else None} for c in selected]

def filter_with_audit(df):
    """Aplica filtros siempre en el orden indicado por la guía y registra exclusiones."""
    rules = [
        ("edad finita y >= 15", F.col("edad").isNotNull() & (F.col("edad") >= 15) & (~F.isnan("edad"))),
        ("ocupado y asalariado (P05C16 en 1,2,3,4)", (F.col("OCUPADOS_n") == 1) & F.col("categoria_ocupacional").isin("1", "2", "3", "4")),
        ("salario finito y > 0", F.col("salario_mensual").isNotNull() & (F.col("salario_mensual") > 0) & (~F.isnan("salario_mensual"))),
        ("componentes de antigüedad válidos", F.col("antiguedad_anios").isNotNull() & (F.col("antiguedad_anios") >= 0) & F.col("antiguedad_meses").isNotNull() & (F.col("antiguedad_meses") >= 0) & (F.col("antiguedad_meses") <= 11) & (F.col("antiguedad_meses") == F.floor("antiguedad_meses"))),
        ("antigüedad <= edad", F.col("antiguedad") <= F.col("edad")),
        ("horas en (0,168]", F.col("horas_semanales").isNotNull() & (F.col("horas_semanales") > 0) & (F.col("horas_semanales") <= 168) & (~F.isnan("horas_semanales"))),
    ]
    audit, current = [], df
    for label, condition in rules:
        before = current.count(); current = current.filter(condition); after = current.count()
        audit.append({"paso": label, "antes": before, "excluidos": before - after, "despues": after})
    return current, audit

def describe(df):
    result = {}
    for c in NUMERIC:
        s = df.select(F.count(c).alias("n"), F.mean(c).alias("media"), F.stddev_samp(c).alias("desv_est"), F.min(c).alias("min"), F.max(c).alias("max")).first().asDict()
        q = df.approxQuantile(c, [0.25, 0.5, 0.75, 0.95], 0.001)
        result[c] = {**{k: float(v) if v is not None else None for k, v in s.items()}, "p25": q[0], "mediana": q[1], "p75": q[2], "p95": q[3]}
    return result

def correlation(df):
    cols = ["salario_mensual", "edad", "antiguedad", "horas_semanales"]
    vec = VectorAssembler(inputCols=cols, outputCol="vector_correlacion", handleInvalid="skip").transform(df)
    matrix = Correlation.corr(vec, "vector_correlacion", "pearson").head()[0].toArray().tolist()
    return {"etiquetas": cols, "matriz": matrix}

def clustering(df):
    # Se excluye salario: el perfil debe describir rasgos laborales, no reproducir directamente el objetivo.
    features = ["edad", "antiguedad", "horas_semanales"]
    assembled = VectorAssembler(inputCols=features, outputCol="cluster_raw", handleInvalid="skip").transform(df)
    scaled = StandardScaler(inputCol="cluster_raw", outputCol="cluster_features", withMean=True, withStd=True).fit(assembled).transform(assembled).cache()
    evaluator = ClusteringEvaluator(featuresCol="cluster_features", predictionCol="cluster", metricName="silhouette", distanceMeasure="squaredEuclidean")
    trials, models = [], {}
    for k in (2, 3, 4, 5):
        model = KMeans(k=k, seed=2026, featuresCol="cluster_features", predictionCol="cluster").fit(scaled)
        pred = model.transform(scaled); score = evaluator.evaluate(pred)
        trials.append({"k": k, "silhouette": score, "wssse": model.summary.trainingCost}); models[k] = (model, pred)
    best = max(trials, key=lambda x: x["silhouette"]); prof = models[best["k"]][1].groupBy("cluster").agg(F.count("*").alias("n"), *[F.avg(c).alias("media_" + c) for c in features + ["salario_mensual"]]).orderBy("cluster")
    return trials, best, prof

def regression_pipeline(reg_param: float, elastic_net: float):
    indexers = [StringIndexer(inputCol=c, outputCol=c + "_idx", handleInvalid="keep") for c in CATEGORICAL]
    encoder = OneHotEncoder(inputCols=[c + "_idx" for c in CATEGORICAL], outputCols=[c + "_oh" for c in CATEGORICAL], handleInvalid="keep")
    assembler = VectorAssembler(inputCols=["edad", "antiguedad", "horas_semanales"] + [c + "_oh" for c in CATEGORICAL], outputCol="features_raw", handleInvalid="keep")
    scaler = StandardScaler(inputCol="features_raw", outputCol="features", withMean=False, withStd=True)
    lr = LinearRegression(featuresCol="features", labelCol="salario_mensual", predictionCol="prediction", regParam=reg_param, elasticNetParam=elastic_net, maxIter=200)
    return Pipeline(stages=indexers + [encoder, assembler, scaler, lr])

def metrics(pred):
    return {name: RegressionEvaluator(labelCol="salario_mensual", predictionCol="prediction", metricName=name).evaluate(pred) for name in ("mae", "rmse", "r2")}

def run(data_dir: str, output_dir: str):
    spark = SparkSession.builder.appName("Lab7-Ejercicios1a5").getOrCreate(); spark.sparkContext.setLogLevel("WARN")
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    sources = discover_sources(data_dir)
    raw = {p: harmonize(read_source(spark, f, p)) for p, f in sources.items()}
    before = [{"periodo": p, "registros": df.count()} for p, df in raw.items()]
    prepared, audit = {}, {}
    for p, df in raw.items(): prepared[p], audit[p] = filter_with_audit(df)
    train25 = prepared["2025T1"].unionByName(prepared["2025T2"]).unionByName(prepared["2025T3"]).unionByName(prepared["2025T4"])
    test26 = prepared["2026T1"]
    train25.write.mode("overwrite").parquet(str(out / "personas_2025_preparado.parquet")); test26.write.mode("overwrite").parquet(str(out / "personas_2026_preparado.parquet"))
    key_cols = ["periodo_archivo", "NUM_HOGAR", "NUM_PERSONA"]
    fingerprint = F.sha2(F.to_json(F.struct(*[F.col(c) for c in train25.columns])), 256)
    duplicates = train25.withColumn("_firma", fingerprint).groupBy(*key_cols).agg(F.count("*").alias("repeticiones"), F.countDistinct("_firma").alias("versiones")).filter("repeticiones > 1")
    duplicate_summary = duplicates.agg(F.count("*").alias("claves_repetidas"), F.sum(F.when(F.col("versiones") == 1, 1).otherwise(0)).alias("repeticiones_exactas"), F.sum(F.when(F.col("versiones") > 1, 1).otherwise(0)).alias("claves_en_conflicto")).first().asDict()
    e1 = {"esquema": {f.name: f.dataType.simpleString() for f in raw["2025T1"].schema.fields}, "muestra_cinco_registros": [json.loads(r) for r in raw["2025T1"].limit(5).toJSON().collect()], "registros_antes": before, "registros_despues": [{"periodo": p, "registros": prepared[p].count()} for p in PERIODS], "faltantes_antes_filtros": missing_table(raw["2025T1"].unionByName(raw["2025T2"]).unionByName(raw["2025T3"]).unionByName(raw["2025T4"])), "auditoria_filtros": audit, "duplicados_2025": {k: int(v or 0) for k, v in duplicate_summary.items()}}
    e2 = {"descriptivos": describe(train25), "por_categoria": {c: [r.asDict() for r in train25.groupBy(c).agg(F.count("*").alias("n"), F.expr("percentile_approx(salario_mensual, 0.5)").alias("salario_mediano")).orderBy(F.desc("n")).collect()] for c in CATEGORICAL}, "por_trimestre": [r.asDict() for r in train25.groupBy("periodo_archivo").agg(F.count("*").alias("n"), F.expr("percentile_approx(salario_mensual, 0.5)").alias("salario_mediano")).orderBy("periodo_archivo").collect()]}
    e3 = correlation(train25)
    trials, best_k, profile = clustering(train25); e4 = {"pruebas": trials, "mejor": best_k, "perfiles": [r.asDict() for r in profile.collect()]}
    train, valid = train25.randomSplit([0.8, 0.2], seed=2026)
    candidates, best_model = [], None
    for reg, elastic in ((0.0, 0.0), (0.1, 0.0), (0.1, 0.5)):
        model = regression_pipeline(reg, elastic).fit(train); value = metrics(model.transform(valid)); row = {"regParam": reg, "elasticNetParam": elastic, **value}; candidates.append(row)
        if best_model is None or row["rmse"] < best_model[0]["rmse"]: best_model = (row, model)
    best_model[1].write().overwrite().save(str(out / "mejor_regresion_lineal"))
    baseline = valid.select(F.avg("salario_mensual").alias("base")).first()["base"]
    baseline_pred = valid.withColumn("prediction", F.lit(baseline)); e5 = {"configuraciones": candidates, "mejor": best_model[0], "referencia_media": metrics(baseline_pred), "particion": {"entrenamiento": train.count(), "validacion": valid.count()}}
    report = {"ejercicio_1": e1, "ejercicio_2": e2, "ejercicio_3": e3, "ejercicio_4": e4, "ejercicio_5": e5}
    (out / "resultados_ejercicios_1_5.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--data-dir", required=True); parser.add_argument("--output-dir", default="salida_lab7")
    args = parser.parse_args(); print(json.dumps(run(args.data_dir, args.output_dir), ensure_ascii=False, indent=2, default=float))
