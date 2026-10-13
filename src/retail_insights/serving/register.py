"""Log the agentic assistant to MLflow, register it in Unity Catalog, optionally deploy.

    retail-register-agent --catalog workspace --schema retail_dev --warehouse-id <id> \\
        [--deploy true]

Steps:
  1. log serving/agent_model.py as a "models from code" ResponsesAgent, with the
     retail_insights package attached via code_paths and every Databricks
     resource it touches declared, so Model Serving can mint credentials for them;
  2. register the logged model as <catalog>.<schema>.retail_agentic_assistant;
  3. with --deploy true, call databricks.agents.deploy: a Model Serving endpoint
     plus the Review App and inference tables.

Runs as a job task (resources/jobs.yml: retail_agent_register). Deploying
creates a serving endpoint that bills while it is up; scale_to_zero is on.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from retail_insights.config import CHAT_ENDPOINT, EMBEDDING_ENDPOINT, Layout

log = logging.getLogger("retail_insights.serving")

PIP_REQUIREMENTS = [
    "mlflow>=3.11",
    "databricks-sdk>=0.60.0",
    "openai>=1.40.0",
    "sqlglot>=25.0.0",
    "langgraph>=0.6.0",
]


def resources_for(lay: Layout, warehouse_id: str, chat_endpoint: str):
    from mlflow.models.resources import (
        DatabricksFunction,
        DatabricksServingEndpoint,
        DatabricksSQLWarehouse,
        DatabricksTable,
        DatabricksVectorSearchIndex,
    )

    tables = [
        "gold_daily_sales",
        "gold_product_performance",
        "gold_customer_rfm",
        "silver_sales",
        "silver_products",
        "silver_stores",
        "gold_filings_meta",
    ]
    return [
        DatabricksServingEndpoint(endpoint_name=chat_endpoint),
        DatabricksServingEndpoint(endpoint_name=EMBEDDING_ENDPOINT),
        DatabricksVectorSearchIndex(index_name=lay.vector_index),
        DatabricksSQLWarehouse(warehouse_id=warehouse_id),
        DatabricksFunction(function_name=lay.filing_lookup_fn),
        *[DatabricksTable(table_name=lay.table(t)) for t in tables],
    ]


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--catalog", required=True)
    p.add_argument("--schema", required=True)
    p.add_argument("--warehouse-id", required=True)
    p.add_argument("--chat-endpoint", default=CHAT_ENDPOINT)
    p.add_argument("--model-name", default="retail_agentic_assistant")
    p.add_argument("--experiment", default="/Shared/databricks-agents-build-deploy/agent")
    p.add_argument("--deploy", default="false")
    p.add_argument("--smoke-question", default="What is our return window for Electronics?")
    args, _ = p.parse_known_args(argv)

    import mlflow

    import retail_insights
    from retail_insights.serving import agent_model

    lay = Layout(args.catalog, args.schema)
    uc_name = lay.table(args.model_name)
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_experiment(args.experiment)

    config = {
        "catalog": args.catalog,
        "schema": args.schema,
        "warehouse_id": args.warehouse_id,
        "chat_endpoint": args.chat_endpoint,
        "retriever": "vector",
    }
    example = {"input": [{"role": "user", "content": args.smoke_question}]}

    with mlflow.start_run(run_name="register-agentic-assistant"):
        info = mlflow.pyfunc.log_model(
            name="agent",
            python_model=agent_model.__file__,
            code_paths=[str(Path(retail_insights.__file__).parent)],
            model_config=config,
            input_example=example,
            resources=resources_for(lay, args.warehouse_id, args.chat_endpoint),
            pip_requirements=PIP_REQUIREMENTS,
        )
        log.info("logged %s", info.model_uri)

        # Smoke test the artifact exactly as serving will load it.
        loaded = mlflow.pyfunc.load_model(info.model_uri)
        answer = loaded.predict(example)
        log.info("smoke answer: %s", str(answer)[:400])

    version = mlflow.register_model(info.model_uri, uc_name)
    log.info("registered %s version %s", uc_name, version.version)

    if args.deploy.lower() == "true":
        from databricks import agents

        deployment = agents.deploy(uc_name, int(version.version), scale_to_zero=True)
        log.info("deploying to endpoint %s", getattr(deployment, "endpoint_name", deployment))


if __name__ == "__main__":
    main()
