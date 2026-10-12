"""Runs the first steps of the docs pipeline on a local SparkSession against a
temporary folder standing in for the Unity Catalog volume.

Covers the parts that only exist inside Spark: the binaryFile reader, the
sidecar join on file stems, and the parsing UDF. MERGE needs Delta Lake, which
is not available locally, so the first-run (create table) path is what runs here.
"""

import json

import pytest
from pyspark.sql import functions as F

from retail_insights.config import Layout
from retail_insights.knowledge import jobs
from tests.test_knowledge import sample_filing_html

pytestmark = pytest.mark.spark


@pytest.fixture
def lay(spark, tmp_path, monkeypatch):
    monkeypatch.setattr(Layout, "volume_path", property(lambda self: str(tmp_path / "volume")))
    spark.sql("CREATE SCHEMA IF NOT EXISTS retail_test")
    for t in ("bronze_documents", "silver_documents"):
        spark.sql(f"DROP TABLE IF EXISTS spark_catalog.retail_test.{t}")
    lay = Layout("spark_catalog", "retail_test")
    folder = tmp_path / "volume" / "landing" / "docs" / "filings" / "EXR"
    folder.mkdir(parents=True)
    (folder / "0001.html").write_text(sample_filing_html())
    (folder / "0001.json").write_text(
        json.dumps(
            {
                "ticker": "EXR",
                "company": "Example Retail Corp",
                "form": "10-K",
                "filing_date": "2026-03-01",
                "accession_number": "0001",
                "url": "https://example.com/f",
            }
        )
    )
    return lay


def test_land_bronze_silver(spark, lay):
    args = ["--catalog", lay.catalog, "--schema", lay.schema]
    jobs.land_policies(args)
    jobs.bronze_docs(args)

    bronze = spark.table(lay.bronze_documents)
    assert bronze.count() == 11
    filing = bronze.where("source = 'filing'").first()
    assert json.loads(filing.sidecar_json)["ticker"] == "EXR"
    assert bronze.where("source = 'policy' AND sidecar_json IS NULL").count() == 10

    jobs.silver_docs(args)
    silver = spark.table(lay.silver_documents)
    assert silver.count() == 11
    row = silver.where("source = 'filing'").first()
    assert row.doc_id == "EXR-0001" and row.metadata["company"] == "Example Retail Corp"
    assert "EntityCentralIndexKey" not in row.text
    assert silver.where(F.col("doc_id") == "POL-PRC-003").first().title == "Pricing and Promotions Playbook"

    # A second run with nothing changed parses nothing new.
    jobs.silver_docs(args)
    assert spark.table(lay.silver_documents).count() == 11


def test_chunks_frame_matches_python_chunker(spark, lay):
    from retail_insights.knowledge.corpus import policy_chunks

    args = ["--catalog", lay.catalog, "--schema", lay.schema]
    jobs.land_policies(args)
    jobs.bronze_docs(args)
    jobs.silver_docs(args)
    chunks = jobs.chunks_frame(spark.table(lay.silver_documents))
    assert set(chunks.columns) >= {*jobs.CHUNK_COLUMNS, "doc_text_hash", "_chunked_at"}
    policy_ids = {r.chunk_id for r in chunks.where("source = 'policy'").select("chunk_id").collect()}
    assert policy_ids == {c.chunk_id for c in policy_chunks()}  # Spark UDF == local chunker
    filing = chunks.where("source = 'filing'").collect()
    assert filing and all(r.ticker == "EXR" and "Item 1" in r.section for r in filing)
