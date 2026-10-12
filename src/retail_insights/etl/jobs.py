"""Job entrypoints. Each function is a console script in pyproject.toml and a
`python_wheel_task` in resources/jobs.yml.

These are the only places that do I/O. All business logic lives in
transforms.py and quality.py.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, timedelta

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from retail_insights.config import REFERENCE_ENTITIES, Layout, add_layout_args, layout_from_args
from retail_insights.etl import datagen, transforms
from retail_insights.etl.quality import SALES_RULES, apply_rules, rule_failure_counts
from retail_insights.etl.schemas import RAW_SALES_SCHEMA, raw_reference_schema

log = logging.getLogger("retail_insights")


def _spark() -> SparkSession:
    return SparkSession.builder.getOrCreate()


def _parse(argv: list[str] | None, extra=None) -> argparse.Namespace:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    add_layout_args(p)
    if extra:
        extra(p)
    args, unknown = p.parse_known_args(argv)
    if unknown:
        log.warning("ignoring unknown args: %s", unknown)
    return args


def _overwrite(df: DataFrame, table: str) -> None:
    df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(table)


# ----------------------------------------------------------------------------- setup
def setup(argv: list[str] | None = None) -> None:
    """Create schema + volume. Idempotent, so it runs at the start of every job."""
    args = _parse(argv)
    lay = layout_from_args(args)
    spark = _spark()
    spark.sql(
        f"CREATE SCHEMA IF NOT EXISTS {lay.catalog}.{lay.schema} COMMENT 'Retail insights medallion tables'"
    )
    spark.sql(
        f"CREATE VOLUME IF NOT EXISTS {lay.catalog}.{lay.schema}.{lay.volume} "
        "COMMENT 'Landing zone and stream checkpoints'"
    )
    log.info("schema and volume ready at %s", lay.volume_path)


# -------------------------------------------------------------------------- generate
def generate(argv: list[str] | None = None) -> None:
    """Land synthetic CSV files in the volume, as an upstream system would."""

    def extra(p):
        p.add_argument("--days", type=int, default=30)
        p.add_argument("--end-date", default="", help="YYYY-MM-DD, default yesterday")
        p.add_argument("--orders-per-day", type=int, default=120)

    args = _parse(argv, extra)
    lay = layout_from_args(args)
    end = date.fromisoformat(args.end_date) if args.end_date else date.today() - timedelta(days=1)
    start = end - timedelta(days=args.days - 1)

    dims = datagen.build_dimensions()
    datagen.land_reference_data(lay.landing_reference, dims)
    files = datagen.land_sales(lay.landing_sales, start, args.days, dims, args.orders_per_day)
    log.info("landed %d sales files (%s .. %s) into %s", len(files), start, end, lay.landing_sales)


# ---------------------------------------------------------------------------- bronze
def bronze(argv: list[str] | None = None) -> None:
    """Raw -> bronze. Reference data is a small full snapshot (batch overwrite).
    Sales are incremental: Auto Loader picks up only files it has not seen,
    tracked in a checkpoint inside the volume."""
    args = _parse(argv)
    lay = layout_from_args(args)
    spark = _spark()

    for entity in REFERENCE_ENTITIES:
        raw = (
            spark.read.option("header", True)
            .schema(raw_reference_schema(entity))
            .csv(f"{lay.landing_reference}/{entity}/")
            .withColumn("_source_file", F.col("_metadata.file_path"))
        )
        _overwrite(transforms.with_ingest_metadata(raw), lay.bronze_reference(entity))

    stream = (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", "csv")
        .option("header", True)
        .option("cloudFiles.schemaLocation", lay.checkpoint("bronze_sales_schema"))
        .option("rescuedDataColumn", "_rescued_data")
        .schema(RAW_SALES_SCHEMA)
        .load(lay.landing_sales)
        .withColumn("_source_file", F.col("_metadata.file_path"))
    )
    query = (
        transforms.with_ingest_metadata(stream)
        .writeStream.option("checkpointLocation", lay.checkpoint("bronze_sales"))
        .option("mergeSchema", "true")
        .trigger(availableNow=True)
        .toTable(lay.bronze_sales)
    )
    query.awaitTermination()
    log.info("bronze_sales now has %d rows", spark.table(lay.bronze_sales).count())


# ---------------------------------------------------------------------------- silver
def _high_water_mark(spark: SparkSession, lay: Layout):
    marks = []
    for t in (lay.silver_sales, lay.quarantine_sales):
        if spark.catalog.tableExists(t):
            marks.append(spark.table(t).agg(F.max("_ingested_at")).first()[0])
    marks = [m for m in marks if m is not None]
    return max(marks) if marks else None


def silver(argv: list[str] | None = None) -> None:
    """Bronze -> silver: type, dedupe, validate, quarantine, MERGE (upsert)."""
    args = _parse(argv)
    lay = layout_from_args(args)
    spark = _spark()

    products = transforms.clean_products(spark.table(lay.bronze_reference("products")))
    _overwrite(products, lay.silver_dim("products"))
    _overwrite(
        transforms.clean_stores(spark.table(lay.bronze_reference("stores"))),
        lay.silver_dim("stores"),
    )
    _overwrite(
        transforms.clean_customers(spark.table(lay.bronze_reference("customers"))),
        lay.silver_dim("customers"),
    )

    # Only rows ingested since the last silver run (simple incremental pattern).
    hwm = _high_water_mark(spark, lay)
    new_bronze = spark.table(lay.bronze_sales)
    if hwm is not None:
        new_bronze = new_bronze.where(F.col("_ingested_at") > F.lit(hwm))

    typed = transforms.clean_sales(new_bronze)
    deduped = transforms.dedupe_latest(typed, ["order_id", "line_no"])
    enriched = transforms.enrich_with_products(deduped, spark.table(lay.silver_dim("products")))
    valid, quarantined = apply_rules(enriched, SALES_RULES)
    valid = valid.drop("_product_known")

    n_in, n_valid, n_bad = typed.count(), valid.count(), quarantined.count()
    log.info(
        "silver batch: in=%d dupes_removed=%d valid=%d quarantined=%d failures=%s",
        n_in,
        n_in - deduped.count(),
        n_valid,
        n_bad,
        rule_failure_counts(quarantined),
    )

    if n_bad:
        quarantined.write.mode("append").option("mergeSchema", "true").saveAsTable(lay.quarantine_sales)

    if not spark.catalog.tableExists(lay.silver_sales):
        valid.write.saveAsTable(lay.silver_sales)
        return

    valid.createOrReplaceTempView("silver_sales_updates")
    spark.sql(f"""
        MERGE INTO {lay.silver_sales} AS t
        USING silver_sales_updates AS s
          ON t.order_id = s.order_id AND t.line_no = s.line_no
        WHEN MATCHED AND s._ingested_at > t._ingested_at THEN UPDATE SET *
        WHEN NOT MATCHED THEN INSERT *
    """)


# ------------------------------------------------------------------------------ gold
TABLE_DOCS = {
    "gold_daily_sales": (
        "Daily sales per store and channel. Revenue counts COMPLETED order lines only; "
        "returns_amount is the value of RETURNED lines. Money columns are USD.",
        {
            "order_date": "Calendar day of the order",
            "channel": "STORE or ONLINE",
            "orders": "Distinct completed orders",
            "net_revenue": "Gross revenue minus discounts",
            "gross_margin": "net_revenue minus cost",
            "avg_order_value": "net_revenue / orders",
        },
    ),
    "gold_product_performance": (
        "Lifetime performance per product: units, revenue, margin, return rate, rank in category.",
        {
            "margin_pct": "gross_margin / net_revenue, 0..1",
            "return_rate": "returned lines / all lines, 0..1",
            "rank_in_category": "1 = highest net revenue within its category",
        },
    ),
    "gold_customer_rfm": (
        "Recency/Frequency/Monetary scores per known customer (guests excluded) with a segment "
        "label: Champions, Loyal, New or Promising, At Risk, Needs Attention.",
        {
            "recency_days": "Days since last completed order",
            "frequency": "Distinct completed orders",
            "monetary": "Total net spend USD",
            "segment": "Plain-English RFM segment",
        },
    ),
}


def _document(spark: SparkSession, table: str, comment: str, cols: dict[str, str]) -> None:
    """Table and column comments in Unity Catalog. The agent reads these through its
    describe_table tool, so good comments directly improve answer quality."""
    esc = lambda s: s.replace("'", "\\'")  # noqa: E731
    spark.sql(f"COMMENT ON TABLE {table} IS '{esc(comment)}'")
    for col, text in cols.items():
        spark.sql(f"ALTER TABLE {table} ALTER COLUMN {col} COMMENT '{esc(text)}'")


def gold(argv: list[str] | None = None) -> None:
    """Silver -> gold. Full recompute: the data is small and it keeps the logic obvious.
    At real scale you would make these incremental (MERGE by date, or materialized views)."""
    args = _parse(argv)
    lay = layout_from_args(args)
    spark = _spark()

    sales = spark.table(lay.silver_sales)
    stores = spark.table(lay.silver_dim("stores"))
    products = spark.table(lay.silver_dim("products"))
    customers = spark.table(lay.silver_dim("customers"))

    outputs = {
        "gold_daily_sales": transforms.gold_daily_sales(sales, stores),
        "gold_product_performance": transforms.gold_product_performance(sales, products),
        "gold_customer_rfm": transforms.gold_customer_rfm(sales, customers),
    }
    for name, df in outputs.items():
        table = lay.table(name)
        _overwrite(df, table)
        comment, cols = TABLE_DOCS[name]
        _document(spark, table, comment, cols)
        log.info("wrote %s", table)


# ---------------------------------------------------------------------- quality gate
def quality_gate(argv: list[str] | None = None) -> None:
    """Fail the job (and so block downstream consumers) if the data looks wrong."""

    def extra(p):
        p.add_argument("--max-quarantine-rate", type=float, default=0.05)
        p.add_argument("--run-id", default="local")

    args = _parse(argv, extra)
    lay = layout_from_args(args)
    spark = _spark()

    silver_rows = spark.table(lay.silver_sales).count()
    bad_rows = (
        spark.table(lay.quarantine_sales).count() if spark.catalog.tableExists(lay.quarantine_sales) else 0
    )
    rate = bad_rows / max(silver_rows + bad_rows, 1)

    silver_net = (
        spark.table(lay.silver_sales).where("status = 'COMPLETED'").agg(F.sum("net_amount")).first()[0] or 0
    )
    gold_net = spark.table(lay.gold_daily_sales).agg(F.sum("net_revenue")).first()[0] or 0

    checks = {
        "silver_not_empty": silver_rows > 0,
        "quarantine_rate_ok": rate <= args.max_quarantine_rate,
        "gold_reconciles_with_silver": abs(float(silver_net) - float(gold_net)) < 0.01,
    }
    result = {
        "run_id": args.run_id,
        "silver_rows": silver_rows,
        "quarantined_rows": bad_rows,
        "quarantine_rate": round(rate, 4),
        "silver_net": float(silver_net),
        "gold_net": float(gold_net),
        "checks": checks,
        "passed": all(checks.values()),
    }
    log.info("quality gate: %s", json.dumps(result))

    spark.createDataFrame(
        [{**{k: v for k, v in result.items() if k != "checks"}, "checks": json.dumps(checks)}]
    ).withColumn("logged_at", F.current_timestamp()).write.mode("append").option(
        "mergeSchema", "true"
    ).saveAsTable(lay.ops_run_log)

    if not result["passed"]:
        failed = [k for k, ok in checks.items() if not ok]
        sys.exit(f"quality gate failed: {failed}")


if __name__ == "__main__":  # handy for `python -m retail_insights.etl.jobs generate ...`
    {
        "setup": setup,
        "generate": generate,
        "bronze": bronze,
        "silver": silver,
        "gold": gold,
        "quality_gate": quality_gate,
    }[sys.argv[1]](sys.argv[2:])
