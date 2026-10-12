TARGET ?= dev
Q ?= Did we give Electronics discounts above what our pricing policy allows?

.PHONY: install lint fmt test eval build validate deploy platform etl docs grant app agent-eval register demo all

install:        ## create .venv with runtime + dev deps
	uv sync

lint:
	uv run ruff check .
	uv run ruff format --check .

fmt:
	uv run ruff check --fix .
	uv run ruff format .

test:
	uv run pytest --cov=retail_insights --cov-report=term-missing

eval:           ## offline evals, same as CI
	uv run retail-eval --offline
	uv run retail-eval-rag --offline

build:
	uv build --wheel

validate:
	databricks bundle validate -t $(TARGET)

deploy:
	databricks bundle deploy -t $(TARGET)

platform:       ## ETL ‖ docs, then evals
	databricks bundle run retail_platform -t $(TARGET)

etl:
	databricks bundle run retail_etl -t $(TARGET)

docs:
	databricks bundle run retail_docs -t $(TARGET)

grant:
	uv run retail-grant-app --app-name retail-ai-$(TARGET) --catalog workspace --schema retail_$(TARGET)

app:
	databricks bundle run retail_app -t $(TARGET)

agent-eval:
	databricks bundle run retail_agent_eval -t $(TARGET)

register:
	databricks bundle run retail_agent_register -t $(TARGET)

demo:           ## stream the agentic graph node by node (needs warehouse + CLI auth)
	uv run python scripts/demo_agentic_trace.py "$(Q)"

all: lint test eval deploy platform grant app   ## everything, end to end
