---
doc_id: POL-SEG-005
title: Customer Segmentation Playbook
owner: Marketing Analytics
effective_date: 2026-04-01
version: 1.4
---

# Customer Segmentation Playbook

## The RFM model

Cedarline Markets segments known customers with an RFM model built from completed orders. Recency is the number of days since the last completed order. Frequency is the number of distinct completed orders. Monetary is total net spend in USD. Each measure is scored from 1 to 5 using quintiles, where 5 is best. Guest checkouts are excluded because they cannot be linked to a person.

The model is refreshed every time the data pipeline runs and is published in the gold customer RFM table.

## Segments

- Champions: recency score 4 or 5 and frequency score 4 or 5. Recent, frequent buyers.
- Loyal: recency score 3 or more and frequency score 3 or more, but not Champions.
- New or Promising: recency score 4 or 5 and frequency score 1 or 2. Recently acquired or reactivated.
- At Risk: recency score 1 or 2 and frequency score 3 or more. Used to buy often but have gone quiet.
- Needs Attention: everyone else.

## Actions by segment

### Champions

Do not discount. Invite to early access events and product previews, and ask for reviews. Champions should receive at most one marketing email per week.

### Loyal

Offer tier upgrade nudges when a customer is within 150 USD of the next loyalty tier. Cross-sell categories the customer has not bought from yet.

### New or Promising

Send the second-purchase series: a welcome email on day 1, category recommendations on day 7 and a 10% offer on day 21 if there is no second order yet.

### At Risk

Run the win-back program: a personalised email within 7 days of the customer entering the segment with a 15% offer valid for 14 days, followed by a reminder after 7 days. If there is no purchase within 30 days, move the customer to the low-frequency newsletter. The win-back offer counts as a promotion and must respect category maximums.

### Needs Attention

Include in general seasonal campaigns only. No individual offers.

## Measuring success

Segment programs are measured by migration between segments month over month. The target is to move at least 20% of At Risk customers back to Loyal or Champions within a quarter.
