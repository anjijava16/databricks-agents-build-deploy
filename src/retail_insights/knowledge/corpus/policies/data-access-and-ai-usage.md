---
doc_id: POL-AI-010
title: Data Access and AI Assistant Usage Policy
owner: Data Governance Office
effective_date: 2026-06-01
version: 1.1
---

# Data Access and AI Assistant Usage Policy

## Purpose

This policy describes how employees may use the retail insights assistant and other AI tools on company data.

## What the assistant can access

The assistant reads curated gold tables (daily sales, product performance and customer RFM segments), selected silver reference tables, the internal policy library and public SEC filings of peer retailers. It runs under its own service identity with read-only access. It cannot change data, send emails or issue refunds.

## Personal data

Customer emails and names are not available to the assistant. Customer-level answers use customer IDs, loyalty tier and home state only. Exporting customer-level lists from the assistant for marketing use requires approval from the Data Governance Office.

## Accuracy and citations

Every number the assistant gives must come from a query it ran, and every policy statement must cite the document it came from. If the assistant cannot ground an answer, it should say so rather than guess. Employees must check figures in the source dashboard before using them in external communication.

## Public filings

Information from competitors' SEC filings is public. It can be used for internal competitive analysis but must be cited with the company, form and filing date.

## Logging

Questions, tool calls and answers are logged for quality monitoring and evaluation. Logs are retained for 180 days and reviewed in aggregate, not to monitor individual employees.
