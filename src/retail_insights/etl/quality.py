"""A small, explicit data-quality layer.

Each rule is a named boolean SQL expression that must be TRUE for a good row.
Rows failing any rule go to a quarantine table together with the list of rules
they failed, instead of being silently dropped. That gives you an audit trail
and a way to replay rows once the upstream bug is fixed.

This is the same idea as Lakeflow Declarative Pipelines expectations
(`@dlt.expect_or_drop`), written in plain PySpark so it runs in a normal job
and in local unit tests.
"""

from __future__ import annotations

from dataclasses import dataclass

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


@dataclass(frozen=True)
class Rule:
    name: str
    expr: str  # TRUE means the row is good


SALES_RULES: tuple[Rule, ...] = (
    Rule("order_id_present", "order_id IS NOT NULL AND order_id <> ''"),
    Rule("valid_timestamp", "order_ts IS NOT NULL"),
    Rule("customer_present", "customer_id IS NOT NULL"),
    Rule("positive_quantity", "quantity IS NOT NULL AND quantity > 0"),
    Rule("non_negative_price", "unit_price IS NOT NULL AND unit_price >= 0"),
    Rule("discount_in_range", "discount_pct BETWEEN 0 AND 0.9"),
    Rule("known_status", "status IN ('COMPLETED', 'RETURNED', 'CANCELLED')"),
    Rule("known_product", "_product_known"),
)


def apply_rules(df: DataFrame, rules: tuple[Rule, ...]) -> tuple[DataFrame, DataFrame]:
    """Split df into (valid, quarantined). Quarantined rows get `_failed_rules`."""
    failed = F.filter(
        F.array(*[F.when(~F.coalesce(F.expr(r.expr), F.lit(False)), F.lit(r.name)) for r in rules]),
        lambda x: x.isNotNull(),
    )
    flagged = df.withColumn("_failed_rules", failed)
    valid = flagged.where(F.size("_failed_rules") == 0).drop("_failed_rules")
    quarantined = flagged.where(F.size("_failed_rules") > 0).withColumn(
        "_quarantined_at", F.current_timestamp()
    )
    return valid, quarantined


def rule_failure_counts(quarantined: DataFrame) -> dict[str, int]:
    rows = quarantined.select(F.explode("_failed_rules").alias("rule")).groupBy("rule").count().collect()
    return {r["rule"]: r["count"] for r in rows}
