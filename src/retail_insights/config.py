"""Names of everything the pipeline reads and writes, derived from catalog + schema.

Keeping every table name in one place means the ETL jobs, the agent tools and
the tests can never drift apart on what a table is called.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class Layout:
    catalog: str
    schema: str
    volume: str = "raw"

    # ---- Unity Catalog names -------------------------------------------------
    def table(self, name: str) -> str:
        return f"{self.catalog}.{self.schema}.{name}"

    @property
    def volume_path(self) -> str:
        return f"/Volumes/{self.catalog}/{self.schema}/{self.volume}"

    # Landing zone (files) inside the volume
    @property
    def landing_sales(self) -> str:
        return f"{self.volume_path}/landing/sales"

    @property
    def landing_reference(self) -> str:
        return f"{self.volume_path}/landing/reference"

    def checkpoint(self, stream: str) -> str:
        return f"{self.volume_path}/_checkpoints/{stream}"

    # Bronze: raw strings + ingest metadata
    @property
    def bronze_sales(self) -> str:
        return self.table("bronze_sales")

    def bronze_reference(self, entity: str) -> str:
        return self.table(f"bronze_{entity}")

    # Silver: typed, deduplicated, validated
    @property
    def silver_sales(self) -> str:
        return self.table("silver_sales")

    @property
    def quarantine_sales(self) -> str:
        return self.table("quarantine_sales")

    def silver_dim(self, entity: str) -> str:
        return self.table(f"silver_{entity}")

    # Gold: business aggregates the agent reads
    @property
    def gold_daily_sales(self) -> str:
        return self.table("gold_daily_sales")

    @property
    def gold_product_performance(self) -> str:
        return self.table("gold_product_performance")

    @property
    def gold_customer_rfm(self) -> str:
        return self.table("gold_customer_rfm")

    @property
    def ops_run_log(self) -> str:
        return self.table("ops_pipeline_runs")

    # ---- Unstructured side: documents for RAG ---------------------------------
    @property
    def landing_docs(self) -> str:
        return f"{self.volume_path}/landing/docs"

    @property
    def bronze_documents(self) -> str:
        return self.table("bronze_documents")

    @property
    def silver_documents(self) -> str:
        return self.table("silver_documents")

    @property
    def doc_chunks(self) -> str:
        """Vector Search source table (Change Data Feed on)."""
        return self.table("gold_doc_chunks")

    @property
    def filings_meta(self) -> str:
        return self.table("gold_filings_meta")

    @property
    def vector_index(self) -> str:
        return self.table("gold_doc_chunks_index")

    @property
    def filing_lookup_fn(self) -> str:
        """UC SQL table function the agent calls for structured filing lookups."""
        return self.table("filing_lookup")


REFERENCE_ENTITIES = ("customers", "products", "stores")

# Shared defaults. All overridable through job parameters / env vars.
VECTOR_SEARCH_ENDPOINT = "retail-ai-vs"
EMBEDDING_ENDPOINT = "databricks-gte-large-en"
CHAT_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"


def add_layout_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--volume", default="raw")


def layout_from_args(args: argparse.Namespace) -> Layout:
    return Layout(catalog=args.catalog, schema=args.schema, volume=args.volume)
