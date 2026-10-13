"""Docs pipeline entry points (console scripts -> python_wheel_tasks in resources/jobs.yml).

    land_policies -> fetch_filings -> bronze_docs -> silver_docs -> chunk_docs
                  -> uc_functions -> vector_index -> docs_quality_gate

Same medallion idea as the sales ETL, applied to documents:

  bronze_documents   one row per file, raw content + hash (binaryFile reader)
  silver_documents   parsed, cleaned text per document
  gold_doc_chunks    one row per chunk, Change Data Feed ON -> Vector Search source
  gold_filings_meta  structured filing registry behind the filing_lookup() UC function
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from retail_insights.config import (
    EMBEDDING_ENDPOINT,
    VECTOR_SEARCH_ENDPOINT,
    Layout,
    add_layout_args,
    layout_from_args,
)

log = logging.getLogger("retail_insights.docs")

CHUNK_COLUMNS = [
    "chunk_id",
    "doc_id",
    "source",
    "title",
    "section",
    "chunk_index",
    "content",
    "embed_text",
    "word_count",
    "ticker",
    "company",
    "filing_date",
    "url",
]
INDEX_COLUMNS = [c for c in CHUNK_COLUMNS if c != "embed_text"]


def _parse(argv, extra=None) -> argparse.Namespace:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    add_layout_args(p)
    if extra:
        extra(p)
    args, unknown = p.parse_known_args(argv)
    if unknown:
        log.warning("ignoring unknown args: %s", unknown)
    return args


def _spark():
    from pyspark.sql import SparkSession

    return SparkSession.builder.getOrCreate()


# ---------------------------------------------------------------------- land_policies
def land_policies(argv: list[str] | None = None) -> None:
    """Copy the policy library shipped in the wheel into the volume landing zone."""
    from retail_insights.knowledge.corpus import policy_files

    lay = layout_from_args(_parse(argv))
    target = Path(lay.landing_docs) / "policies"
    target.mkdir(parents=True, exist_ok=True)
    files = policy_files()
    for name, text in files.items():
        (target / name).write_text(text, encoding="utf-8")
    log.info("landed %d policy documents in %s", len(files), target)


# ---------------------------------------------------------------------- fetch_filings
def fetch_filings(argv: list[str] | None = None) -> None:
    """Download the latest 10-K for each ticker into landing/docs/filings/<ticker>/.

    Skips filings already on disk. Network errors are logged, not fatal, unless
    --require-filings is set: a workspace without internet egress still gets a
    working policy-only knowledge base."""
    from retail_insights.knowledge.edgar import RETAIL_PEERS, EdgarClient

    def extra(p):
        p.add_argument("--tickers", default=",".join(RETAIL_PEERS))
        p.add_argument("--user-agent", default=os.environ.get("SEC_EDGAR_USER_AGENT", ""))
        p.add_argument("--require-filings", default="false")

    args = _parse(argv, extra)
    lay = layout_from_args(args)
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    if not tickers:
        log.info("no tickers configured, skipping filings")
        return
    if "@" not in args.user_agent:
        msg = "set --user-agent (job parameter sec_user_agent) to 'Name email@domain' for SEC EDGAR"
        if args.require_filings.lower() == "true":
            sys.exit(msg)
        log.warning("%s; skipping filings", msg)
        return

    client = EdgarClient(args.user_agent)
    
    ok, failed = 0, []
    for ticker in tickers:
        try:
            ref = client.latest_filing(ticker)
            folder = Path(lay.landing_docs) / "filings" / ticker
            html_path = folder / f"{ref.accession_number}.html"
            if html_path.exists():
                log.info("%s %s already landed", ticker, ref.accession_number)
                ok += 1
                continue
            folder.mkdir(parents=True, exist_ok=True)
            html_path.write_text(client.download(ref), encoding="utf-8")
            (folder / f"{ref.accession_number}.json").write_text(json.dumps(ref.sidecar()))
            log.info("landed %s %s filed %s", ticker, ref.form, ref.filing_date)
            ok += 1
        except Exception as e:  # network, 404, rate limit
            log.warning("could not fetch %s: %s", ticker, e)
            failed.append(ticker)
    log.info("filings: %d ok, %d failed %s", ok, len(failed), failed)
    if failed and args.require_filings.lower() == "true":
        sys.exit(f"failed to fetch filings for {failed}")


# ------------------------------------------------------------------------ bronze_docs
def bronze_docs(argv: list[str] | None = None) -> None:
    """Every .md/.html file in the landing zone -> one bronze row, upserted by path."""
    from pyspark.sql import functions as F

    lay = layout_from_args(_parse(argv))
    spark = _spark()
    files = (
        spark.read.format("binaryFile")
        .option("recursiveFileLookup", "true")
        .option("pathGlobFilter", "*.{md,html}")
        .load(lay.landing_docs)
    )
    sidecars = _read_sidecars(spark, lay)
    rows = (
        files.select(
            F.regexp_replace("path", "^dbfs:", "").alias("path"),
            F.col("modificationTime").alias("modified_at"),
            F.col("length").alias("size_bytes"),
            F.decode("content", "UTF-8").alias("raw_content"),
        )
        .withColumn("source", F.when(F.col("path").endswith(".md"), "policy").otherwise("filing"))
        .withColumn("content_hash", F.sha2("raw_content", 256))
        .withColumn("_stem", F.regexp_replace("path", r"\.html?$", ""))
        .join(sidecars, "_stem", "left")
        .drop("_stem")
        .withColumn("_ingested_at", F.current_timestamp())
    )
    _upsert(spark, rows, lay.bronze_documents, key="path", changed="content_hash")
    log.info("bronze_documents: %d files", spark.table(lay.bronze_documents).count())


def _read_sidecars(spark, lay: Layout):
    from pyspark.sql import functions as F
    from pyspark.sql.types import StringType, StructField, StructType

    schema = StructType([StructField("_stem", StringType()), StructField("sidecar_json", StringType())])
    try:
        js = (
            spark.read.format("text")
            .option("recursiveFileLookup", "true")
            .option("pathGlobFilter", "*.json")
            .option("wholetext", "true")
            .load(f"{lay.landing_docs}/filings")
        )
    except Exception:  # folder missing (no filings fetched)
        return spark.createDataFrame([], schema)
    return js.select(
        F.regexp_replace(F.regexp_replace(F.col("_metadata.file_path"), "^dbfs:", ""), r"\.json$", "").alias(
            "_stem"
        ),
        F.col("value").alias("sidecar_json"),
    )


def _upsert(spark, df, table: str, key: str, changed: str) -> None:
    if not spark.catalog.tableExists(table):
        df.write.saveAsTable(table)
        return
    view = f"_upd_{table.split('.')[-1]}"
    df.createOrReplaceTempView(view)
    spark.sql(f"""
        MERGE INTO {table} t USING {view} s ON t.{key} = s.{key}
        WHEN MATCHED AND t.{changed} <> s.{changed} THEN UPDATE SET *
        WHEN NOT MATCHED THEN INSERT *
    """)


# ------------------------------------------------------------------------ silver_docs
def silver_docs(argv: list[str] | None = None) -> None:
    """Parse documents whose content changed since the last run (by content_hash)."""
    from pyspark.sql import functions as F
    from pyspark.sql.types import MapType, StringType, StructField, StructType

    from retail_insights.knowledge.documents import parse_file

    lay = layout_from_args(_parse(argv))
    spark = _spark()

    parsed_type = StructType(
        [
            StructField("doc_id", StringType()),
            StructField("title", StringType()),
            StructField("text", StringType()),
            StructField("metadata", MapType(StringType(), StringType())),
            StructField("error", StringType()),
        ]
    )

    # Plain (pickled) Python UDFs: deterministic across runtimes whatever the
    # Arrow-UDF default is, and these rows are few and large, where Arrow buys nothing.
    @F.udf(returnType=parsed_type, useArrow=False)
    def parse(path, content, sidecar):
        try:
            d = parse_file(path, content, sidecar)
            return (d.doc_id, d.title, d.text, {k: str(v) for k, v in d.metadata.items()}, None)
        except Exception as e:  # one bad file must not kill the batch
            return (None, None, None, None, f"{type(e).__name__}: {e}")

    bronze = spark.table(lay.bronze_documents)
    if spark.catalog.tableExists(lay.silver_documents):
        done = spark.table(lay.silver_documents).select("path", F.col("raw_hash").alias("_done"))
        bronze = bronze.join(done, "path", "left").where(
            F.col("_done").isNull() | (F.col("_done") != F.col("content_hash"))
        )

    parsed = (
        bronze.withColumn("p", parse("path", "raw_content", "sidecar_json"))
        .select(
            "path",
            "source",
            F.col("content_hash").alias("raw_hash"),
            "p.doc_id",
            "p.title",
            "p.text",
            "p.metadata",
            "p.error",
        )
        .withColumn("text_hash", F.sha2("text", 256))
        .withColumn("word_count", F.size(F.split("text", r"\s+")))
        .withColumn("_parsed_at", F.current_timestamp())
    )
    errors = parsed.where("error IS NOT NULL").select("path", "error").collect()
    for e in errors:
        log.warning("parse failed: %s -> %s", e.path, e.error)
    good = parsed.where("error IS NULL").drop("error")
    n = good.count()
    log.info("silver_docs: %d new/changed documents, %d parse errors", n, len(errors))
    if n:
        _upsert(spark, good, lay.silver_documents, key="path", changed="raw_hash")


# ------------------------------------------------------------------------- chunk_docs
def chunks_frame(docs):
    """silver_documents rows -> one row per chunk (pure transformation, no I/O)."""
    from pyspark.sql import functions as F
    from pyspark.sql.types import ArrayType, IntegerType, StringType, StructField, StructType

    from retail_insights.knowledge.chunking import chunk_document
    from retail_insights.knowledge.documents import Document

    chunk_type = ArrayType(
        StructType(
            [
                StructField(c, IntegerType() if c in ("chunk_index", "word_count") else StringType())
                for c in CHUNK_COLUMNS
            ]
        )
    )

    @F.udf(returnType=chunk_type, useArrow=False)
    def chunk(doc_id, source, title, text, metadata):
        doc = Document(doc_id, source, title, text or "", dict(metadata or {}))
        return [tuple(c.to_dict()[k] for k in CHUNK_COLUMNS) for c in chunk_document(doc)]

    return (
        docs.select(
            "text_hash",
            F.explode(chunk("doc_id", "source", "title", "text", "metadata")).alias("c"),
        )
        .select("c.*", F.col("text_hash").alias("doc_text_hash"))
        .withColumn("_chunked_at", F.current_timestamp())
    )


def chunk_docs(argv: list[str] | None = None) -> None:
    """silver_documents -> gold_doc_chunks (+ gold_filings_meta).

    MERGE keyed on chunk_id; chunks of a re-parsed document that no longer exist
    are deleted, and Change Data Feed carries inserts, updates and deletes to the
    Vector Search index on the next sync."""
    from pyspark.sql import functions as F

    lay = layout_from_args(_parse(argv))
    spark = _spark()

    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {lay.doc_chunks} (
          chunk_id STRING NOT NULL, doc_id STRING, source STRING, title STRING, section STRING,
          chunk_index INT, content STRING, embed_text STRING, word_count INT, ticker STRING,
          company STRING, filing_date STRING, url STRING, doc_text_hash STRING,
          _chunked_at TIMESTAMP)
        COMMENT 'Document chunks for RAG. Source table of the Vector Search index.'
        TBLPROPERTIES (delta.enableChangeDataFeed = true)
    """)

    docs = spark.table(lay.silver_documents)
    existing = spark.table(lay.doc_chunks).select("doc_id", "doc_text_hash").distinct()
    changed = docs.join(
        existing,
        (docs.doc_id == existing.doc_id) & (docs.text_hash == existing.doc_text_hash),
        "left_anti",
    )
    new_chunks = chunks_frame(changed)
    changed_ids = [r.doc_id for r in changed.select("doc_id").collect()]
    if changed_ids:
        new_chunks.createOrReplaceTempView("_new_chunks")
        id_list = ", ".join("'" + d.replace("'", "''") + "'" for d in changed_ids)
        spark.sql(f"""
            MERGE INTO {lay.doc_chunks} t USING _new_chunks s ON t.chunk_id = s.chunk_id
            WHEN MATCHED THEN UPDATE SET *
            WHEN NOT MATCHED THEN INSERT *
            WHEN NOT MATCHED BY SOURCE AND t.doc_id IN ({id_list}) THEN DELETE
        """)
    log.info("chunk_docs: %d changed documents re-chunked", len(changed_ids))

    # Structured registry of filings for the filing_lookup() UC function.
    meta = docs.where("source = 'filing'").select(
        F.col("metadata")["ticker"].alias("ticker"),
        F.col("metadata")["company"].alias("company"),
        F.col("metadata")["form"].alias("form"),
        F.expr("try_to_date(metadata['filing_date'])").alias("filing_date"),
        F.col("metadata")["accession_number"].alias("accession_number"),
        F.col("metadata")["url"].alias("url"),
        "word_count",
    )
    meta.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(lay.filings_meta)
    spark.sql(
        f"COMMENT ON TABLE {lay.filings_meta} IS 'Registry of SEC filings in the knowledge base, "
        "one row per filing. Backs the filing_lookup() function.'"
    )


# ----------------------------------------------------------------------- uc_functions
def uc_functions(argv: list[str] | None = None) -> None:
    """Register Unity Catalog functions the agents call as governed tools."""
    lay = layout_from_args(_parse(argv))
    spark = _spark()
    spark.sql(f"""
        CREATE OR REPLACE FUNCTION {lay.filing_lookup_fn}(p_ticker STRING)
        RETURNS TABLE (ticker STRING, company STRING, form STRING, filing_date DATE,
                       accession_number STRING, url STRING)
        COMMENT 'Filings in the knowledge base. Pass a ticker such as WMT, or an empty string for all.'
        RETURN SELECT ticker, company, form, filing_date, accession_number, url
               FROM {lay.filings_meta}
               WHERE coalesce(p_ticker, '') = '' OR upper(ticker) = upper(p_ticker)
    """)
    log.info("registered %s", lay.filing_lookup_fn)


# ----------------------------------------------------------------------- vector_index
def vector_index(argv: list[str] | None = None) -> None:
    """Make sure the Vector Search endpoint and Delta Sync index exist, then sync.

    Done in a job task rather than as bundle resources on purpose: a Delta Sync
    index can only be created once its source table exists, which is after the
    first pipeline run, not at `bundle deploy` time."""

    def extra(p):
        p.add_argument("--endpoint", default=VECTOR_SEARCH_ENDPOINT)
        p.add_argument("--embedding-endpoint", default=EMBEDDING_ENDPOINT)
        p.add_argument("--timeout-min", type=int, default=40)

    args = _parse(argv, extra)
    lay = layout_from_args(args)
    from databricks.sdk import WorkspaceClient

    from retail_insights.rag.index_admin import ensure_endpoint, ensure_index, wait_until_synced

    ws = WorkspaceClient()
    ensure_endpoint(ws, args.endpoint)
    created = ensure_index(
        ws,
        index_name=lay.vector_index,
        endpoint=args.endpoint,
        source_table=lay.doc_chunks,
        embedding_endpoint=args.embedding_endpoint,
        columns=INDEX_COLUMNS,
    )
    if not created:
        ws.vector_search_indexes.sync_index(lay.vector_index)
    expected = _spark().table(lay.doc_chunks).count()
    wait_until_synced(ws, lay.vector_index, expected, timeout_s=args.timeout_min * 60)


# ------------------------------------------------------------------ docs_quality_gate
SMOKE_QUERIES = [
    ("How many days do customers have to return electronics?", "POL-RET-001"),
    ("What is the maximum promotional discount for Grocery?", "POL-PRC-003"),
    ("What should we do with At Risk customers?", "POL-SEG-005"),
]


def docs_quality_gate(argv: list[str] | None = None) -> None:
    """Fail the run if the knowledge base or the index looks wrong."""

    def extra(p):
        p.add_argument("--run-id", default="local")
        p.add_argument("--skip-index", default="false")

    args = _parse(argv, extra)
    lay = layout_from_args(args)
    spark = _spark()
    from pyspark.sql import functions as F

    chunks = spark.table(lay.doc_chunks)
    stats = chunks.agg(
        F.count("*").alias("n"),
        F.countDistinct("chunk_id").alias("unique_ids"),
        F.countDistinct("doc_id").alias("docs"),
        F.sum(F.when(F.length(F.trim("content")) == 0, 1).otherwise(0)).alias("empty"),
        F.max("word_count").alias("max_words"),
    ).first()
    policies = {r.doc_id for r in chunks.where("source = 'policy'").select("doc_id").distinct().collect()}
    checks = {
        "chunks_exist": stats.n > 0,
        "chunk_ids_unique": stats.n == stats.unique_ids,
        "no_empty_chunks": stats.empty == 0,
        "chunk_size_bounded": (stats.max_words or 0) <= 400,
        "all_policies_indexed": len(policies) >= 10,
    }
    retrieval = {}
    if args.skip_index.lower() != "true":
        from retail_insights.rag.retrieval import VectorSearchRetriever

        retriever = VectorSearchRetriever(lay.vector_index)
        for q, _expected in SMOKE_QUERIES:
            hits = retriever.search(q, k=3)
            retrieval[q] = [h.doc_id for h in hits]
        checks["smoke_retrieval_top3"] = all(exp in retrieval[q] for q, exp in SMOKE_QUERIES)
    result = {
        "run_id": args.run_id,
        "chunks": stats.n,
        "documents": stats.docs,
        "checks": checks,
        "retrieval": retrieval,
        "passed": all(checks.values()),
    }
    log.info("docs quality gate: %s", json.dumps(result))
    spark.createDataFrame(
        [
            {
                "run_id": args.run_id,
                "pipeline": "docs",
                "passed": result["passed"],
                "detail": json.dumps(result),
            }
        ]
    ).withColumn("logged_at", F.current_timestamp()).write.mode("append").option(
        "mergeSchema", "true"
    ).saveAsTable(lay.table("ops_docs_runs"))
    if not result["passed"]:
        sys.exit(f"docs quality gate failed: {[k for k, v in checks.items() if not v]}")
