import os
import shutil
import sys
import tempfile

import pytest

# MLflow (used by the serving tests) would otherwise create ./mlflow.db and print
# agent hints; keep both out of the repo.
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
# Spark's Python UDF workers must use this interpreter, not whatever `python3` is on PATH.
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("MLFLOW_TRACKING_URI", f"sqlite:///{tempfile.mkdtemp()}/mlflow.db")


@pytest.fixture(scope="session")
def spark():
    if shutil.which("java") is None:
        pytest.skip("Java not installed; Spark tests skipped")
    from pyspark.sql import SparkSession

    session = (
        SparkSession.builder.master("local[1]")
        .appName("retail-ai-tests")
        .config("spark.sql.shuffle.partitions", "1")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.ansi.enabled", "true")  # match Databricks serverless
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.warehouse.dir", tempfile.mkdtemp(prefix="spark-warehouse-"))
        .getOrCreate()
    )
    yield session
    session.stop()
