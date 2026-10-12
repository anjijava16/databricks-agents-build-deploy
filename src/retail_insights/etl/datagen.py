"""Synthetic retail data, deterministic for a given seed.

Plain Python on purpose: no Spark needed, so it is trivial to unit test and it
runs anywhere. It writes CSV files into the landing zone the same way an
upstream POS export or an SFTP drop would.

The generator deliberately injects a small amount of dirty data (duplicates,
missing customers, negative quantities, unknown products, bad timestamps) so
the silver-layer quality rules have something real to catch.
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path

DIM_SEED = 7  # dimensions never change between runs, so FKs stay valid

REGIONS = {
    "Northeast": [("Philadelphia", "PA"), ("New York", "NY"), ("Boston", "MA")],
    "South": [("Atlanta", "GA"), ("Austin", "TX")],
    "Midwest": [("Chicago", "IL"), ("Columbus", "OH")],
    "West": [("Seattle", "WA"), ("Denver", "CO")],
}

CATALOG = {
    "Grocery": [
        ("Organic Bananas", 0.79),
        ("Whole Milk 1gal", 4.29),
        ("Sourdough Loaf", 5.49),
        ("Free Range Eggs 12ct", 4.99),
        ("Cold Brew Coffee", 6.99),
        ("Greek Yogurt", 1.89),
        ("Basmati Rice 5lb", 9.49),
        ("Olive Oil 1L", 12.99),
    ],
    "Electronics": [
        ("Wireless Earbuds", 59.99),
        ("USB-C Charger 65W", 34.99),
        ("Bluetooth Speaker", 79.99),
        ("Smart Plug 2pk", 24.99),
        ("Fitness Tracker", 99.99),
        ("Phone Case", 19.99),
    ],
    "Home": [
        ("Bath Towel Set", 39.99),
        ("Scented Candle", 14.99),
        ("Cast Iron Skillet", 29.99),
        ("Storage Bins 3pk", 22.99),
        ("LED Desk Lamp", 27.99),
        ("Throw Blanket", 34.99),
    ],
    "Apparel": [
        ("Crew Socks 6pk", 16.99),
        ("Rain Jacket", 89.99),
        ("Running Shorts", 29.99),
        ("Cotton Tee", 12.99),
        ("Denim Jeans", 49.99),
        ("Wool Beanie", 18.99),
    ],
    "Beauty": [
        ("Daily Moisturizer", 18.49),
        ("SPF 50 Sunscreen", 13.99),
        ("Shampoo", 8.99),
        ("Lip Balm 3pk", 5.99),
        ("Face Wash", 10.49),
    ],
}

# Margin by category (cost = price * (1 - margin))
MARGINS = {"Grocery": 0.28, "Electronics": 0.35, "Home": 0.45, "Apparel": 0.55, "Beauty": 0.6}

SALES_COLUMNS = [
    "order_id",
    "line_no",
    "order_ts",
    "customer_id",
    "store_id",
    "channel",
    "product_id",
    "quantity",
    "unit_price",
    "discount_pct",
    "payment_type",
    "status",
]


@dataclass
class Dimensions:
    stores: list[dict] = field(default_factory=list)
    products: list[dict] = field(default_factory=list)
    customers: list[dict] = field(default_factory=list)


def build_dimensions(n_customers: int = 500) -> Dimensions:
    rng = random.Random(DIM_SEED)
    dims = Dimensions()

    sid = 1
    for region, cities in REGIONS.items():
        for city, state in cities:
            dims.stores.append(
                {
                    "store_id": f"S{sid:03d}",
                    "store_name": f"{city} Market",
                    "city": city,
                    "state": state,
                    "region": region,
                    "opened_date": date(2015 + rng.randint(0, 8), rng.randint(1, 12), 1).isoformat(),
                }
            )
            sid += 1

    pid = 1
    for category, items in CATALOG.items():
        for name, price in items:
            dims.products.append(
                {
                    "product_id": f"P{pid:04d}",
                    "sku": f"SKU-{category[:3].upper()}-{pid:04d}",
                    "product_name": name,
                    "category": category,
                    "unit_price": f"{price:.2f}",
                    "unit_cost": f"{price * (1 - MARGINS[category]):.2f}",
                }
            )
            pid += 1

    first_names = [
        "Ava",
        "Liam",
        "Mia",
        "Noah",
        "Zoe",
        "Ethan",
        "Priya",
        "Arjun",
        "Sofia",
        "Mateo",
        "Chloe",
        "Lucas",
        "Aisha",
        "Omar",
        "Grace",
        "Leo",
        "Nina",
        "Ravi",
    ]
    tiers = ["Bronze"] * 6 + ["Silver"] * 3 + ["Gold"]
    states = [s for cities in REGIONS.values() for _, s in cities]
    for i in range(1, n_customers + 1):
        name = rng.choice(first_names)
        dims.customers.append(
            {
                "customer_id": f"C{i:05d}",
                "first_name": name,
                "email": f"{name.lower()}.{i}@example.com",
                "signup_date": (date(2022, 1, 1) + timedelta(days=rng.randint(0, 1000))).isoformat(),
                "loyalty_tier": rng.choice(tiers),
                "home_state": rng.choice(states),
            }
        )
    return dims


def _day_seed(day: date) -> int:
    return int(day.strftime("%Y%m%d"))


def generate_sales(
    day: date, dims: Dimensions, orders_per_day: int = 120, dirty_rate: float = 0.02
) -> list[dict]:
    """Order lines for one calendar day. Same day => same rows (idempotent re-runs)."""
    rng = random.Random(_day_seed(day))
    weekend_boost = 1.35 if day.weekday() >= 5 else 1.0
    n_orders = int(orders_per_day * weekend_boost * rng.uniform(0.85, 1.15))
    rows: list[dict] = []

    for o in range(n_orders):
        order_id = f"O{day.strftime('%Y%m%d')}{o:05d}"
        store = rng.choice(dims.stores)
        channel = "ONLINE" if rng.random() < 0.3 else "STORE"
        customer = rng.choice(dims.customers) if rng.random() < 0.85 else None  # guest checkout
        ts = datetime.combine(day, time(rng.randint(8, 21), rng.randint(0, 59), rng.randint(0, 59)))
        status = rng.choices(["COMPLETED", "RETURNED", "CANCELLED"], weights=[92, 5, 3])[0]
        payment = (
            rng.choice(["CARD", "CARD", "CARD", "WALLET", "CASH"])
            if channel == "STORE"
            else rng.choice(["CARD", "WALLET"])
        )
        promo = rng.random() < 0.2

        for line_no, product in enumerate(rng.sample(dims.products, rng.randint(1, 4)), start=1):
            rows.append(
                {
                    "order_id": order_id,
                    "line_no": str(line_no),
                    "order_ts": ts.isoformat(sep=" "),
                    "customer_id": customer["customer_id"] if customer else "GUEST",
                    "store_id": store["store_id"],
                    "channel": channel,
                    "product_id": product["product_id"],
                    "quantity": str(rng.randint(1, 3)),
                    "unit_price": product["unit_price"],
                    "discount_pct": f"{rng.choice([0.1, 0.15, 0.2]) if promo else 0.0:.2f}",
                    "payment_type": payment,
                    "status": status,
                }
            )

    _inject_dirty_rows(rows, rng, dirty_rate)
    return rows


def _inject_dirty_rows(rows: list[dict], rng: random.Random, rate: float) -> None:
    """Mutate a small share of rows the way real feeds go wrong."""
    if not rows:
        return
    n_bad = max(2, int(len(rows) * rate))
    for i in range(n_bad):
        r = dict(rng.choice(rows))
        problem = rng.choice(["duplicate", "null_customer", "neg_qty", "bad_product", "bad_ts"])
        if problem != "duplicate":
            # A corrupt line gets its own key so dedupe cannot hide it from the quality rules.
            r["line_no"] = str(90 + i)
        if problem == "null_customer":
            r["customer_id"] = ""
        elif problem == "neg_qty":
            r["quantity"] = "-2"
        elif problem == "bad_product":
            r["product_id"] = "P9999"
        elif problem == "bad_ts":
            r["order_ts"] = "not-a-timestamp"
        rows.append(r)  # "duplicate" appends an exact copy


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def land_reference_data(landing_reference: str, dims: Dimensions) -> None:
    base = Path(landing_reference)
    for entity, rows in (
        ("stores", dims.stores),
        ("products", dims.products),
        ("customers", dims.customers),
    ):
        write_csv(base / entity / f"{entity}.csv", rows, list(rows[0].keys()))


def land_sales(
    landing_sales: str, start: date, days: int, dims: Dimensions, orders_per_day: int = 120
) -> list[Path]:
    """One file per day. File name is the day, so re-landing a day overwrites it."""
    written = []
    for i in range(days):
        day = start + timedelta(days=i)
        path = Path(landing_sales) / f"sales_{day.strftime('%Y%m%d')}.csv"
        write_csv(path, generate_sales(day, dims, orders_per_day), SALES_COLUMNS)
        written.append(path)
    return written
