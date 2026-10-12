"""Cross-file consistency checks. They catch the boring breakages that otherwise
only show up after a deploy: an entry point renamed in pyproject but not in a
job, a task depending on a task that doesn't exist, an orchestrator pointing at
an undefined job, an app command pointing at a moved file."""

import itertools
import re
import tomllib
from pathlib import Path

import yaml

from retail_insights.evals.rag_runner import run_offline

ROOT = Path(__file__).resolve().parents[1]
HOST = "https://dbc-d40d7b72-eeb2.cloud.databricks.com"


def load(path: str):
    return yaml.safe_load((ROOT / path).read_text())


def jobs() -> dict:
    return load("resources/jobs.yml")["resources"]["jobs"]


def test_bundle_targets_and_host():
    bundle = load("databricks.yml")
    assert set(bundle["targets"]) == {"dev", "staging", "prod"}
    assert bundle["targets"]["dev"]["default"] is True
    assert bundle["targets"]["prod"]["mode"] == "production"
    for t in bundle["targets"].values():
        assert t["workspace"]["host"] == HOST
        assert t["variables"]["schema"].startswith("retail_")


def test_no_other_cloud_services_in_config():
    text = " ".join(
        (ROOT / p).read_text().lower()
        for p in [
            "databricks.yml",
            "resources/jobs.yml",
            "resources/apps.yml",
            "app.yaml",
            ".github/workflows/ci.yml",
            ".github/workflows/cd.yml",
            "requirements.txt",
        ]
    )
    for word in ("azure", "key vault", "groq", "chromadb", "aws_", "gcp"):
        assert word not in text, word


def test_every_wheel_entry_point_exists_and_imports():
    scripts = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["scripts"]
    used = {
        t["python_wheel_task"]["entry_point"]
        for job in jobs().values()
        for t in job["tasks"]
        if "python_wheel_task" in t
    }
    assert used <= set(scripts), used - set(scripts)
    for target in scripts.values():
        module, func = target.split(":")
        assert callable(getattr(__import__(module, fromlist=[func]), func))


def test_task_dependencies_resolve():
    for name, job in jobs().items():
        keys = {t["task_key"] for t in job["tasks"]}
        for t in job["tasks"]:
            for dep in t.get("depends_on", []):
                assert dep["task_key"] in keys, (name, t["task_key"], dep)


def test_etl_task_graph_is_a_chain():
    tasks = jobs()["retail_etl"]["tasks"]
    assert [t["task_key"] for t in tasks] == ["setup", "generate", "bronze", "silver", "gold", "quality_gate"]
    for prev, task in itertools.pairwise(tasks):
        assert task["depends_on"] == [{"task_key": prev["task_key"]}]


def test_docs_pipeline_fans_out_and_in():
    tasks = {t["task_key"]: t for t in jobs()["retail_docs"]["tasks"]}
    deps = lambda k: {d["task_key"] for d in tasks[k].get("depends_on", [])}  # noqa: E731
    assert deps("land_policies") == deps("fetch_filings") == {"setup"}
    assert deps("bronze_docs") == {"land_policies", "fetch_filings"}
    assert deps("docs_quality_gate") == {"vector_index", "uc_functions"}


def test_orchestrator_runs_known_jobs():
    tasks = {t["task_key"]: t for t in jobs()["retail_platform"]["tasks"]}
    for t in tasks.values():
        ref = re.fullmatch(r"\$\{resources\.jobs\.(\w+)\.id\}", t["run_job_task"]["job_id"])
        assert ref and ref.group(1) in jobs()
    assert {d["task_key"] for d in tasks["evaluate"]["depends_on"]} == {"sales_etl", "documents"}


def test_app_wiring():
    app_yaml = load("app.yaml")
    assert (ROOT / app_yaml["command"][-1]).exists()
    value_from = {e["valueFrom"] for e in app_yaml["env"] if "valueFrom" in e}
    declared = {r["name"] for r in load("resources/apps.yml")["resources"]["apps"]["retail_app"]["resources"]}
    assert value_from <= declared


def test_requirements_cover_runtime_deps():
    deps = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["dependencies"]
    reqs = (ROOT / "requirements.txt").read_text()
    for d in deps:
        assert re.split(r"[<>=\[]", d)[0] in reqs


def test_offline_rag_eval_is_green():
    report = run_offline()
    assert report["passed"], report


def test_workflows_reference_real_jobs_scripts_and_app():
    """Every `bundle run X` in CD must be a job or app in the bundle, every
    `uv run retail-*` in CI/CD must be a console script, and the grant step must
    target the app name the bundle creates."""
    scripts = set(tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["scripts"])
    resources = {**jobs(), **load("resources/apps.yml")["resources"]["apps"]}
    for wf in (".github/workflows/ci.yml", ".github/workflows/cd.yml"):
        text = (ROOT / wf).read_text()
        for key in re.findall(r"databricks bundle run (\w+)", text):
            assert key in resources, (wf, key)
        for script in re.findall(r"uv run (?:--no-dev )?(retail-[\w-]+)", text):
            assert script in scripts, (wf, script)
    app_name = load("resources/apps.yml")["resources"]["apps"]["retail_app"]["name"]
    assert app_name == "retail-ai-${bundle.target}"
    assert '--app-name "retail-ai-$TARGET"' in (ROOT / ".github/workflows/cd.yml").read_text()
    # docs pipeline (creates the VS endpoint) must run before the grant step
    cd = (ROOT / ".github/workflows/cd.yml").read_text()
    assert (
        cd.index("bundle run retail_docs") < cd.index("retail-grant-app") < cd.index("bundle run retail_app")
    )
