# Retail AI Platform on Databricks

A retail data and AI project built end to end on Databricks. It covers four things:

- **ETL.** A sales medallion pipeline (Auto Loader → bronze → silver with quality rules and quarantine → gold), plus a document pipeline that turns internal policies and competitors' SEC 10-K filings into chunks in a Vector Search index.
- **A simple agent.** A tool-calling analytics assistant that answers number questions with guarded SQL on a serverless warehouse.
- **RAG.** Retrieve once, answer with citations, refuse if the answer can't be cited.
- **Agentic RAG.** A LangGraph state machine that plans, routes to SQL, policies, filings or a Unity Catalog function, grades what it retrieved, rewrites failed searches, and verifies citations before answering.

Everything is deployed with a Databricks Asset Bundle (five serverless jobs and a Databricks App), shipped by GitHub Actions, traced and evaluated with MLflow, and optionally served behind Model Serving. No other cloud services are involved.

**Start here: [README_UPDATE.md](README_UPDATE.md)** is the deep dive. It goes from the 50,000-ft view through architecture, flows and theory down to individual tables, functions and YAML keys. The original write-up of the ETL and the analytics agent is in [docs/v1-etl-and-analytics-agent.md](docs/v1-etl-and-analytics-agent.md).

## Quick start

```bash
uv sync
make lint test eval        # no workspace needed: ruff, pytest (incl. local Spark), offline evals

databricks auth login --host https://dbc-d40d7b72-eeb2.cloud.databricks.com
make deploy                # wheel + 5 jobs + app
databricks bundle run retail_platform -t dev --var sec_user_agent="Your Name you@yourdomain.com"
make grant app             # app service principal grants, then start the app
```

Open the app URL that `make app` prints, keep the mode on **Agentic RAG**, and try *"Did we give Electronics discounts above what our pricing policy allows in the last 30 days?"*

## Layout

| Path | What |
|---|---|
| `databricks.yml`, `resources/` | Asset Bundle: variables, targets, jobs, app |
| `src/retail_insights/etl/` | Sales medallion ETL |
| `src/retail_insights/knowledge/` | Policy library, EDGAR client, parsing, chunking, docs pipeline jobs |
| `src/retail_insights/rag/` | Retrievers (Vector Search, BM25), citation gate, classic RAG |
| `src/retail_insights/agentic/` | LangGraph agentic RAG |
| `src/retail_insights/agent/` | Analytics agent, SQL guard, clients, runtime factory |
| `src/retail_insights/evals/` | Golden sets, offline and live evaluation |
| `src/retail_insights/serving/` | MLflow ResponsesAgent, UC registration, deployment |
| `src/retail_insights/app/` | Streamlit Databricks App |
| `.github/workflows/` | CI (no credentials) and CD (manual, per environment) |
