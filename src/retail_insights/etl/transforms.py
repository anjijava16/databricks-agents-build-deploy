"""Pure DataFrame -> DataFrame transformations for every medallion layer.

Nothing here touches storage. The job entrypoints in `jobs.py` do the reading
and writing; these functions only describe the logic. That split is what makes
the pipeline unit-testable on a laptop with a local SparkSession.

Casting uses try_cast / try_to_timestamp because Databricks serverless runs
with ANSI mode on, where a plain cast of "abc" to INT raises instead of
returning NULL. We want NULL so the quality rules can quarantine the row.
"""

from __future__ import annotations

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

MONEY = "DECIMAL(12,2)"


def _blank_to_null(col: str) -> F.Column:
    return F.when(F.trim(F.col(col)) == "", None).otherwise(F.trim(F.col(col)))


def _try_cast(col: str, sql_type: str) -> F.Column:
    return F.expr(f"try_cast(nullif(trim({col}), '') AS {sql_type})")


# --------------------------------------------------------------------------- bronze
def with_ingest_metadata(df: DataFrame, batch_id: str | None = None) -> DataFrame:
    out = df.withColumn("_ingested_at", F.current_timestamp())
    if batch_id is not None:
        out = out.withColumn("_batch_id", F.lit(batch_id))
    return out


# --------------------------------------------------------------------------- silver
def clean_sales(bronze: DataFrame) -> DataFrame:
    """Type, normalise and derive money columns. Bad values become NULL, not errors."""
    return (
        bronze.select(
            _blank_to_null("order_id").alias("order_id"),
            _try_cast("line_no", "INT").alias("line_no"),
            F.expr("try_to_timestamp(nullif(trim(order_ts), ''))").alias("order_ts"),
            _blank_to_null("customer_id").alias("customer_id"),
            _blank_to_null("store_id").alias("store_id"),
            F.upper(_blank_to_null("channel")).alias("channel"),
            _blank_to_null("product_id").alias("product_id"),
            _try_cast("quantity", "INT").alias("quantity"),
            _try_cast("unit_price", MONEY).alias("unit_price"),
            F.coalesce(_try_cast("discount_pct", "DECIMAL(4,2)"), F.lit(0).cast("DECIMAL(4,2)")).alias(
                "discount_pct"
            ),
            F.upper(_blank_to_null("payment_type")).alias("payment_type"),
            F.upper(_blank_to_null("status")).alias("status"),
            F.col("_ingested_at"),
            F.col("_source_file"),
        )
        .withColumn("order_date", F.to_date("order_ts"))
        .withColumn("gross_amount", (F.col("quantity") * F.col("unit_price")).cast(MONEY))
        .withColumn("discount_amount", F.round(F.col("gross_amount") * F.col("discount_pct"), 2).cast(MONEY))
        .withColumn("net_amount", (F.col("gross_amount") - F.col("discount_amount")).cast(MONEY))
    )


def dedupe_latest(df: DataFrame, keys: list[str], order_col: str = "_ingested_at") -> DataFrame:
    """Keep one row per business key, the most recently ingested one."""
    w = Window.partitionBy(*keys).orderBy(F.col(order_col).desc())
    return df.withColumn("_rn", F.row_number().over(w)).where("_rn = 1").drop("_rn")


def enrich_with_products(sales: DataFrame, products: DataFrame) -> DataFrame:
    """Attach category and unit cost, and flag lines whose product is unknown."""
    p = products.select(
        "product_id",
        "category",
        F.col("unit_cost").alias("_unit_cost"),
        F.lit(True).alias("_product_known"),
    )
    return (
        sales.join(p, "product_id", "left")
        .withColumn("_product_known", F.coalesce("_product_known", F.lit(False)))
        .withColumn("cost_amount", (F.col("quantity") * F.col("_unit_cost")).cast(MONEY))
        .drop("_unit_cost")
    )


def clean_products(raw: DataFrame) -> DataFrame:
    return (
        raw.select(
            _blank_to_null("product_id").alias("product_id"),
            "sku",
            "product_name",
            "category",
            _try_cast("unit_price", MONEY).alias("unit_price"),
            _try_cast("unit_cost", MONEY).alias("unit_cost"),
        )
        .where("product_id IS NOT NULL")
        .dropDuplicates(["product_id"])
    )


def clean_stores(raw: DataFrame) -> DataFrame:
    return (
        raw.select(
            _blank_to_null("store_id").alias("store_id"),
            "store_name",
            "city",
            "state",
            "region",
            _try_cast("opened_date", "DATE").alias("opened_date"),
        )
        .where("store_id IS NOT NULL")
        .dropDuplicates(["store_id"])
    )


def clean_customers(raw: DataFrame) -> DataFrame:
    return (
        raw.select(
            _blank_to_null("customer_id").alias("customer_id"),
            "first_name",
            F.lower(F.trim("email")).alias("email"),
            _try_cast("signup_date", "DATE").alias("signup_date"),
            "loyalty_tier",
            "home_state",
        )
        .where("customer_id IS NOT NULL")
        .dropDuplicates(["customer_id"])
    )


# --------------------------------------------------------------------------- gold
def gold_daily_sales(sales: DataFrame, stores: DataFrame) -> DataFrame:
    """One row per day x store x channel. Revenue counts COMPLETED lines only;
    returns are reported separately so net revenue is never double counted."""
    completed = F.col("status") == "COMPLETED"
    returned = F.col("status") == "RETURNED"
    agg = sales.groupBy("order_date", "store_id", "channel").agg(
        F.countDistinct(F.when(completed, F.col("order_id"))).alias("orders"),
        F.sum(F.when(completed, F.col("quantity")).otherwise(0)).alias("units"),
        F.sum(F.when(completed, F.col("gross_amount")).otherwise(0)).alias("gross_revenue"),
        F.sum(F.when(completed, F.col("discount_amount")).otherwise(0)).alias("discounts"),
        F.sum(F.when(completed, F.col("net_amount")).otherwise(0)).alias("net_revenue"),
        F.sum(F.when(completed, F.col("cost_amount")).otherwise(0)).alias("cost"),
        F.sum(F.when(returned, F.col("net_amount")).otherwise(0)).alias("returns_amount"),
    )
    return (
        agg.join(stores.select("store_id", "store_name", "region"), "store_id", "left")
        .withColumn("gross_margin", (F.col("net_revenue") - F.col("cost")).cast(MONEY))
        .withColumn(
            "avg_order_value",
            F.when(F.col("orders") > 0, F.round(F.col("net_revenue") / F.col("orders"), 2)).cast(MONEY),
        )
        .select(
            "order_date",
            "store_id",
            "store_name",
            "region",
            "channel",
            "orders",
            "units",
            "gross_revenue",
            "discounts",
            "net_revenue",
            "cost",
            "gross_margin",
            "returns_amount",
            "avg_order_value",
        )
    )


def gold_product_performance(sales: DataFrame, products: DataFrame) -> DataFrame:
    completed = F.col("status") == "COMPLETED"
    agg = sales.groupBy("product_id").agg(
        F.sum(F.when(completed, F.col("quantity")).otherwise(0)).alias("units_sold"),
        F.sum(F.when(completed, F.col("net_amount")).otherwise(0)).alias("net_revenue"),
        F.sum(F.when(completed, F.col("cost_amount")).otherwise(0)).alias("cost"),
        F.count(F.when(F.col("status") == "RETURNED", 1)).alias("returned_lines"),
        F.count(F.lit(1)).alias("total_lines"),
    )
    w = Window.partitionBy("category").orderBy(F.col("net_revenue").desc())
    return (
        agg.join(products.select("product_id", "product_name", "category"), "product_id")
        .withColumn("gross_margin", (F.col("net_revenue") - F.col("cost")).cast(MONEY))
        .withColumn(
            "margin_pct",
            F.round(F.when(F.col("net_revenue") > 0, F.col("gross_margin") / F.col("net_revenue")), 4),
        )
        .withColumn("return_rate", F.round(F.col("returned_lines") / F.col("total_lines"), 4))
        .withColumn("rank_in_category", F.dense_rank().over(w))
        .select(
            "product_id",
            "product_name",
            "category",
            "units_sold",
            "net_revenue",
            "gross_margin",
            "margin_pct",
            "return_rate",
            "rank_in_category",
        )
    )


def gold_customer_rfm(sales: DataFrame, customers: DataFrame, as_of: str | None = None) -> DataFrame:
    """Recency / Frequency / Monetary scoring with quintiles, plus a plain-English
    segment label the agent can talk about. Guest checkouts are excluded."""
    base = sales.where((F.col("status") == "COMPLETED") & (F.col("customer_id") != "GUEST"))
    per_cust = base.groupBy("customer_id").agg(
        F.max("order_date").alias("last_order_date"),
        F.countDistinct("order_id").alias("frequency"),
        F.sum("net_amount").cast(MONEY).alias("monetary"),
    )
    # Default "today" is the latest order in the data, so scores are stable on re-runs.
    as_of_col = F.lit(as_of).cast("date") if as_of else F.max("last_order_date").over(Window.partitionBy())
    per_cust = per_cust.withColumn("_as_of", as_of_col)

    scored = (
        per_cust.withColumn("recency_days", F.datediff("_as_of", "last_order_date"))
        .withColumn("r_score", F.ntile(5).over(Window.orderBy(F.col("recency_days").desc())))
        .withColumn("f_score", F.ntile(5).over(Window.orderBy("frequency")))
        .withColumn("m_score", F.ntile(5).over(Window.orderBy("monetary")))
    )
    segment = (
        F.when((F.col("r_score") >= 4) & (F.col("f_score") >= 4), "Champions")
        .when((F.col("r_score") >= 3) & (F.col("f_score") >= 3), "Loyal")
        .when((F.col("r_score") >= 4) & (F.col("f_score") <= 2), "New or Promising")
        .when((F.col("r_score") <= 2) & (F.col("f_score") >= 3), "At Risk")
        .otherwise("Needs Attention")
    )
    return (
        scored.withColumn("segment", segment)
        .join(customers.select("customer_id", "loyalty_tier", "home_state"), "customer_id", "left")
        .select(
            "customer_id",
            "loyalty_tier",
            "home_state",
            "last_order_date",
            "recency_days",
            "frequency",
            "monetary",
            "r_score",
            "f_score",
            "m_score",
            "segment",
        )
    )
