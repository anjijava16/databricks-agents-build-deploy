from datetime import date

from retail_insights.etl import datagen


def test_dimensions_are_stable_and_unique():
    a, b = datagen.build_dimensions(), datagen.build_dimensions()
    assert a.products == b.products and a.customers == b.customers
    assert len({p["product_id"] for p in a.products}) == len(a.products)
    assert len(a.stores) == 9
    assert all(float(p["unit_cost"]) < float(p["unit_price"]) for p in a.products)


def test_sales_are_deterministic_per_day():
    dims = datagen.build_dimensions()
    d = date(2026, 3, 14)
    assert datagen.generate_sales(d, dims) == datagen.generate_sales(d, dims)
    assert datagen.generate_sales(d, dims) != datagen.generate_sales(date(2026, 3, 15), dims)


def test_sales_contain_dirty_rows_for_quality_rules():
    dims = datagen.build_dimensions()
    rows = [r for i in range(10) for r in datagen.generate_sales(date(2026, 1, 1 + i), dims)]
    assert any(r["quantity"].startswith("-") for r in rows)
    assert any(r["product_id"] == "P9999" for r in rows)
    keys = [(r["order_id"], r["line_no"]) for r in rows]
    assert len(keys) != len(set(keys)), "expected duplicate order lines"
    assert set(rows[0]) == set(datagen.SALES_COLUMNS)


def test_land_sales_writes_one_file_per_day(tmp_path):
    dims = datagen.build_dimensions()
    files = datagen.land_sales(str(tmp_path / "sales"), date(2026, 2, 1), 3, dims, orders_per_day=5)
    assert [f.name for f in files] == [
        "sales_20260201.csv",
        "sales_20260202.csv",
        "sales_20260203.csv",
    ]
    header = files[0].read_text().splitlines()[0]
    assert header.split(",") == datagen.SALES_COLUMNS
    datagen.land_reference_data(str(tmp_path / "ref"), dims)
    assert (tmp_path / "ref" / "products" / "products.csv").exists()
