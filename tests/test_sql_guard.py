import pytest

from retail_insights.agent.sql_guard import SQLPolicy, UnsafeSQLError, guard_sql
from retail_insights.agent.tools import AGENT_TABLES
from retail_insights.evals.runner import RED_TEAM_SQL

POLICY = SQLPolicy("workspace", "retail_dev", AGENT_TABLES, max_rows=100)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT region, SUM(net_revenue) FROM gold_daily_sales GROUP BY region",
        "select * from workspace.retail_dev.gold_product_performance where category = 'Home'",
        "WITH t AS (SELECT * FROM silver_sales) SELECT COUNT(*) FROM t",
        "SELECT store_id FROM gold_daily_sales UNION SELECT store_id FROM silver_stores",
        "SELECT 'DROP TABLE x' AS harmless_string FROM silver_products",
    ],
)
def test_allows_read_queries(sql):
    out = guard_sql(sql, POLICY)
    assert "LIMIT" in out.upper()


@pytest.mark.parametrize("sql", RED_TEAM_SQL)
def test_blocks_red_team(sql):
    with pytest.raises(UnsafeSQLError):
        guard_sql(sql, POLICY)


def test_unknown_table_is_rejected_with_hint():
    with pytest.raises(UnsafeSQLError, match="list_tables"):
        guard_sql("SELECT * FROM payroll", POLICY)


def test_adds_and_tightens_limit():
    assert guard_sql("SELECT * FROM silver_sales", POLICY).endswith("LIMIT 100")
    assert guard_sql("SELECT * FROM silver_sales LIMIT 5", POLICY).endswith("LIMIT 5")
    assert guard_sql("SELECT * FROM silver_sales LIMIT 100000", POLICY).endswith("LIMIT 100")


def test_rejects_garbage_and_empty():
    for bad in ["", "   ", "SELEC * FRM"]:
        with pytest.raises(UnsafeSQLError):
            guard_sql(bad, POLICY)


def test_customer_pii_table_is_not_queryable():
    with pytest.raises(UnsafeSQLError, match="not available"):
        guard_sql("SELECT email, first_name FROM silver_customers", POLICY)
