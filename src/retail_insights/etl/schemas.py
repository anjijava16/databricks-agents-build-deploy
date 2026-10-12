"""Explicit schemas. Bronze reads everything as string so a malformed value can
never fail ingestion; typing happens in silver where bad values are caught."""

from pyspark.sql.types import StringType, StructField, StructType

from retail_insights.etl.datagen import SALES_COLUMNS

RAW_SALES_SCHEMA = StructType([StructField(c, StringType(), True) for c in SALES_COLUMNS])

RAW_REFERENCE_COLUMNS = {
    "stores": ["store_id", "store_name", "city", "state", "region", "opened_date"],
    "products": ["product_id", "sku", "product_name", "category", "unit_price", "unit_cost"],
    "customers": [
        "customer_id",
        "first_name",
        "email",
        "signup_date",
        "loyalty_tier",
        "home_state",
    ],
}


def raw_reference_schema(entity: str) -> StructType:
    return StructType([StructField(c, StringType(), True) for c in RAW_REFERENCE_COLUMNS[entity]])
