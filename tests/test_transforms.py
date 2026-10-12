"""Medallion logic on a local SparkSession with hand-built rows, so every
expected number below can be checked by hand."""

from datetime import date
from decimal import Decimal

import pytest
from pyspark.sql import functions as F

from retail_insights.etl import transforms
from retail_insights.etl.quality import SALES_RULES, apply_rules, rule_failure_counts
from retail_insights.etl.schemas import RAW_SALES_SCHEMA

pytestmark = pytest.mark.spark


def _bronze(spark, rows):
    cols = RAW_SALES_SCHEMA.fieldNames()
    df = spark.createDataFrame([tuple(r.get(c) for c in cols) for r in rows], RAW_SALES_SCHEMA)
    return transforms.with_ingest_metadata(df).withColumn("_source_file", F.lit("test.csv"))


def line(
    order_id="O1",
    line_no="1",
    qty="2",
    price="10.00",
    disc="0.10",
    status="COMPLETED",
    product="P0001",
    customer="C00001",
    ts="2026-01-05 10:00:00",
    channel="store",
    store="S001",
    payment="card",
):
    return {
        "order_id": order_id,
        "line_no": line_no,
        "order_ts": ts,
        "customer_id": customer,
        "store_id": store,
        "channel": channel,
        "product_id": product,
        "quantity": qty,
        "unit_price": price,
        "discount_pct": disc,
        "payment_type": payment,
        "status": status,
    }


@pytest.fixture
def products(spark):
    return spark.createDataFrame(
        [
            ("P0001", "SKU-1", "Bananas", "Grocery", Decimal("10.00"), Decimal("6.00")),
            ("P0002", "SKU-2", "Earbuds", "Electronics", Decimal("50.00"), Decimal("30.00")),
        ],
        "product_id string, sku string, product_name string, category string, "
        "unit_price decimal(12,2), unit_cost decimal(12,2)",
    )


@pytest.fixture
def stores(spark):
    return spark.createDataFrame(
        [("S001", "Philly Market", "Northeast")],
        "store_id string, store_name string, region string",
    )


def test_clean_sales_types_and_money(spark):
    out = transforms.clean_sales(_bronze(spark, [line()])).first()
    assert out.quantity == 2 and out.channel == "STORE" and out.payment_type == "CARD"
    assert out.order_date == date(2026, 1, 5)
    assert out.gross_amount == Decimal("20.00")
    assert out.discount_amount == Decimal("2.00")
    assert out.net_amount == Decimal("18.00")


def test_bad_values_become_null_not_errors(spark):
    # ANSI mode is on (like serverless); a plain cast would raise here.
    out = transforms.clean_sales(_bronze(spark, [line(qty="abc", ts="not-a-timestamp", customer="")])).first()
    assert out.quantity is None and out.order_ts is None and out.customer_id is None


def test_dedupe_keeps_latest_ingest(spark):
    df = spark.createDataFrame(
        [
            ("O1", 1, "2026-01-01 00:00:00", "old"),
            ("O1", 1, "2026-01-02 00:00:00", "new"),
            ("O2", 1, "2026-01-01 00:00:00", "only"),
        ],
        "order_id string, line_no int, _ingested_at string, v string",
    ).withColumn("_ingested_at", F.to_timestamp("_ingested_at"))
    out = {r.order_id: r.v for r in transforms.dedupe_latest(df, ["order_id", "line_no"]).collect()}
    assert out == {"O1": "new", "O2": "only"}


def test_quality_rules_split_and_explain(spark, products):
    rows = [
        line(order_id="GOOD"),
        line(order_id="NEG", qty="-1"),
        line(order_id="UNK", product="P9999"),
        line(order_id="NOCUST", customer=""),
        line(order_id="BADTS", ts="garbage"),
    ]
    enriched = transforms.enrich_with_products(transforms.clean_sales(_bronze(spark, rows)), products)
    valid, bad = apply_rules(enriched, SALES_RULES)

    assert [r.order_id for r in valid.collect()] == ["GOOD"]
    failures = {r.order_id: set(r._failed_rules) for r in bad.collect()}
    assert failures == {
        "NEG": {"positive_quantity"},
        "UNK": {"known_product"},
        "NOCUST": {"customer_present"},
        "BADTS": {"valid_timestamp"},
    }
    assert rule_failure_counts(bad)["known_product"] == 1


def test_enrich_adds_cost(spark, products):
    out = transforms.enrich_with_products(
        transforms.clean_sales(_bronze(spark, [line(qty="3")])), products
    ).first()
    assert out.category == "Grocery" and out.cost_amount == Decimal("18.00")


def _silver(spark, products, rows):
    enriched = transforms.enrich_with_products(transforms.clean_sales(_bronze(spark, rows)), products)
    valid, _ = apply_rules(enriched, SALES_RULES)
    return valid.drop("_product_known")


def test_gold_daily_sales_counts_completed_only(spark, products, stores):
    silver = _silver(
        spark,
        products,
        [
            line(order_id="O1", line_no="1", qty="2", disc="0.00"),  # 20 net, 12 cost
            line(order_id="O1", line_no="2", qty="1", price="50.00", product="P0002", disc="0.00"),
            line(order_id="O2", qty="1", disc="0.00", status="RETURNED"),  # return 10
            line(order_id="O3", qty="1", disc="0.00", status="CANCELLED"),
        ],
    )
    g = transforms.gold_daily_sales(silver, stores).first()
    assert g.orders == 1 and g.units == 3
    assert g.net_revenue == Decimal("70.00")
    assert g.cost == Decimal("42.00") and g.gross_margin == Decimal("28.00")
    assert g.returns_amount == Decimal("10.00")
    assert g.avg_order_value == Decimal("70.00")
    assert g.region == "Northeast"


def test_gold_product_performance_rank_and_rates(spark, products):
    silver = _silver(
        spark,
        products,
        [
            line(order_id="A", qty="1", disc="0.00"),
            line(order_id="B", qty="1", disc="0.00", status="RETURNED"),
            line(order_id="C", qty="1", price="50.00", product="P0002", disc="0.00"),
        ],
    )
    rows = {r.product_id: r for r in transforms.gold_product_performance(silver, products).collect()}
    assert float(rows["P0001"].return_rate) == pytest.approx(0.5)
    assert float(rows["P0001"].margin_pct) == pytest.approx(0.4)
    assert rows["P0002"].rank_in_category == 1


def test_gold_rfm_excludes_guests_and_labels_segments(spark, products):
    customers = spark.createDataFrame(
        [(f"C{i}", "Gold", "PA") for i in range(1, 6)],
        "customer_id string, loyalty_tier string, home_state string",
    )
    rows = [
        line(order_id=f"O{i}{j}", customer=f"C{i}", ts=f"2026-01-{10 + i:02d} 10:00:00")
        for i in range(1, 6)
        for j in range(i)
    ]
    rows.append(line(order_id="G1", customer="GUEST"))
    rfm = transforms.gold_customer_rfm(_silver(spark, products, rows), customers)
    out = {r.customer_id: r for r in rfm.collect()}
    assert "GUEST" not in out and len(out) == 5
    assert out["C5"].frequency == 5 and out["C5"].recency_days == 0
    assert out["C5"].segment == "Champions"
    assert out["C1"].recency_days == 4


def test_pipeline_end_to_end_on_generated_files(spark, tmp_path):
    """Generate -> read CSV -> silver -> gold on real synthetic files, then apply
    the same checks the quality_gate job runs on Databricks."""
    from retail_insights.etl import datagen
    from retail_insights.etl.schemas import raw_reference_schema

    dims = datagen.build_dimensions()
    datagen.land_reference_data(str(tmp_path / "ref"), dims)
    datagen.land_sales(str(tmp_path / "sales"), date(2026, 3, 1), 5, dims, orders_per_day=40)

    def read(path, schema):
        return transforms.with_ingest_metadata(
            spark.read.option("header", True).schema(schema).csv(path)
        ).withColumn("_source_file", F.col("_metadata.file_path"))

    products = transforms.clean_products(
        read(str(tmp_path / "ref/products"), raw_reference_schema("products"))
    )
    stores = transforms.clean_stores(read(str(tmp_path / "ref/stores"), raw_reference_schema("stores")))
    typed = transforms.clean_sales(read(str(tmp_path / "sales"), RAW_SALES_SCHEMA))
    deduped = transforms.dedupe_latest(typed, ["order_id", "line_no"])
    valid, bad = apply_rules(transforms.enrich_with_products(deduped, products), SALES_RULES)
    valid = valid.drop("_product_known")

    n_valid, n_bad = valid.count(), bad.count()
    assert n_valid > 300 and 0 < n_bad / (n_valid + n_bad) < 0.05
    assert deduped.count() < typed.count()  # duplicates were injected and removed

    gold = transforms.gold_daily_sales(valid, stores)
    silver_net = valid.where("status = 'COMPLETED'").agg(F.sum("net_amount")).first()[0]
    gold_net = gold.agg(F.sum("net_revenue")).first()[0]
    assert silver_net == gold_net
    assert gold.select("order_date").distinct().count() == 5
