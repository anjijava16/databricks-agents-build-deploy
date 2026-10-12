---
doc_id: POL-GLO-006
title: Data and Metrics Glossary
owner: Data Platform Team
effective_date: 2026-01-15
version: 2.0
---

# Data and Metrics Glossary

## Why this exists

Different teams used to calculate revenue in different ways. This glossary is the single definition used by dashboards, finance reports and the retail insights assistant. The gold tables in the lakehouse implement these definitions.

## Revenue metrics

Gross revenue is quantity multiplied by unit price for completed order lines, before discounts.

Discounts are the money value of promotional discounts on completed lines.

Net revenue is gross revenue minus discounts, for completed order lines only. Returned and cancelled lines are never included in net revenue.

Returns amount is the net value of order lines with status RETURNED. It is reported next to net revenue, never subtracted from it, so it is not double counted.

## Profitability metrics

Cost is quantity multiplied by unit cost for completed lines. Gross margin is net revenue minus cost. Margin percentage is gross margin divided by net revenue, shown as a value between 0 and 1.

## Order metrics

Orders is the count of distinct completed orders. Units is the total quantity on completed lines. Average order value (AOV) is net revenue divided by orders.

## Product metrics

Return rate is the number of returned lines divided by all lines for a product. Rank in category ranks products by net revenue within their category, where 1 is the highest.

## Channels and payment

Channel is STORE for purchases made at a register and ONLINE for web and app orders. Payment type is CARD, WALLET (digital wallets) or CASH. Online orders cannot be paid in cash.

## Data quality

Order lines that fail validation (missing order id, unparseable timestamp, missing customer, non-positive quantity, negative price, discount outside 0 to 90%, unknown status or unknown product) are moved to the quarantine table with the list of rules they failed. They are not counted in any metric until fixed and replayed. The pipeline fails if more than 5% of lines in a run are quarantined.

## Freshness

Sales data lands once a day. The gold tables are rebuilt by the daily pipeline at 6:00 AM Eastern time, so dashboards show data up to the end of the previous day.
