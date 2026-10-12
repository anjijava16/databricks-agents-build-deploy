"""Evaluation for document RAG and agentic RAG.

    retail-eval-rag --offline                       # CI, every pull request
    retail-eval-rag --catalog ... --schema ... --warehouse-id ...   # Databricks job

Offline (no workspace, no tokens):
  * retrieval quality: recall@k and MRR of the real BM25 retriever over the real
    policy library, against hand-labelled expected documents;
  * citation gate contract: grounded drafts pass, uncited or out-of-range drafts
    are refused;
  * routing: the fallback keyword planner sends each cross-source question to the
    expected sources;
  * agentic contract: the full LangGraph runs for every cross-source question with
    a scripted model and a fake warehouse, and must end grounded on the right routes.

Live (Databricks job):
  * the same retrieval metrics against the real Vector Search index;
  * every golden question through the real model: grounded rate, refusal behaviour,
    route accuracy;
  * MLflow GenAI evaluation with Databricks-hosted judges: Correctness (against
    expected facts), RelevanceToQuery, Safety, RetrievalGroundedness, plus our
    deterministic citation scorer. Results land in an MLflow experiment, traces in
    Unity Catalog.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from importlib import resources

from retail_insights.agent.core import RetailAgent
from retail_insights.agent.executor import QueryResult
from retail_insights.agent.fakes import FakeExecutor, RuleLLM, call, say
from retail_insights.agent.tools import Toolbox
from retail_insights.agentic.graph import AgenticRAG, heuristic_plan
from retail_insights.knowledge.corpus import policy_chunks
from retail_insights.rag.chain import REFUSAL, RagChain
from retail_insights.rag.citations import CITATION
from retail_insights.rag.retrieval import BM25Retriever, Retriever


def load_rag_golden() -> list[dict]:
    text = resources.files("retail_insights.evals").joinpath("rag_golden.json").read_text()
    return json.loads(text)


# --------------------------------------------------------------------------- metrics
def retrieval_metrics(
    retriever: Retriever, items: list[dict], k: int = 5, source: str | None = "policy"
) -> dict:
    per = []
    for g in items:
        expected = set(g["expected_docs"])
        if not expected:
            continue
        docs = [p.doc_id for p in retriever.search(g["question"], k=k, source=source)]
        rank = next((i for i, d in enumerate(docs, 1) if d in expected), None)
        per.append({"id": g["id"], "hit": rank is not None, "rank": rank, "top": docs})
    n = len(per) or 1
    return {
        f"recall@{k}": sum(p["hit"] for p in per) / n,
        "mrr": sum(1 / p["rank"] for p in per if p["rank"]) / n,
        "per_question": per,
    }


def route_match(expected: list[str], got: list[str]) -> bool:
    return set(expected) <= set(got)


# --------------------------------------------------------------------------- offline
def _citing_llm(answer_for: str = "") -> RuleLLM:
    return RuleLLM([("policy and research assistant", lambda s, m: answer_for)])


def citation_contract() -> dict:
    retriever = BM25Retriever(policy_chunks())
    q = "How many days do customers have to return electronics?"
    cases = {
        "cited_answer_passes": ("Electronics can be returned within 15 days [1].", True),
        "uncited_answer_refused": ("Electronics can be returned within 15 days.", False),
        "invalid_citation_refused": ("Electronics can be returned within 15 days [9].", False),
    }
    out = {}
    for name, (draft, should_ground) in cases.items():
        r = RagChain(_citing_llm(draft), retriever).run(q)
        out[name] = r.grounded == should_ground and (r.answer == REFUSAL) == (not should_ground)
    return out


def _agentic_fake(g: dict) -> AgenticRAG:
    plan = {
        "tasks": [
            {"route": r, "query": g["question"], "ticker": "BBY" if r != "policies" else None}
            for r in g["routes"]
        ]
    }

    def synth(system, messages):
        evidence = system.split("Evidence:", 1)[-1]
        n = len(re.findall(r"^\[(\d+)\]", evidence, flags=re.MULTILINE))
        cites = "".join(f"[{i}]" for i in range(1, n + 1))
        return f"The evidence answers this question directly {cites}."

    def analyst(system, messages):
        if messages[-1]["role"] == "tool":
            return say("There were 42 matching order lines.")
        return call("run_sql", query="SELECT COUNT(*) AS v FROM silver_sales")

    llm = RuleLLM(
        [
            ("You plan how to answer", lambda s, m: json.dumps(plan)),
            ("You check whether retrieved passages", lambda s, m: '{"relevant": [1, 2]}'),
            ("Answer the user's question using ONLY the numbered evidence", synth),
            ("retail analytics assistant", analyst),
            ("A search over", lambda s, m: "rewritten query"),
        ]
    )
    ex = FakeExecutor(lambda sql: QueryResult(["v"], [[42]]))
    return AgenticRAG(
        llm,
        BM25Retriever(policy_chunks()),
        analytics=RetailAgent(llm, Toolbox(ex, "workspace", "retail_dev")),
        executor=ex,
        filing_lookup_fn="workspace.retail_dev.filing_lookup",
    )


def run_offline() -> dict:
    golden = load_rag_golden()
    policy_items = [g for g in golden if g.get("kind") != "agentic"]
    agentic_items = [g for g in golden if g.get("kind") == "agentic" and not g.get("expect_refusal")]

    retrieval = retrieval_metrics(BM25Retriever(policy_chunks()), policy_items, k=5)
    routing = []
    for g in agentic_items:
        got = [t.route for t in heuristic_plan(g["question"])]
        routing.append(
            {"id": g["id"], "expected": g["routes"], "got": got, "ok": route_match(g["routes"], got)}
        )
    agentic = []
    for g in agentic_items:
        if "filings" in g["routes"]:
            continue  # no filings in the offline corpus
        r = _agentic_fake(g).run(g["question"])
        agentic.append(
            {
                "id": g["id"],
                "grounded": r.grounded,
                "routes": r.routes,
                "ok": r.grounded and route_match(g["routes"], r.routes),
            }
        )
    report = {
        "mode": "offline",
        "retrieval": retrieval,
        "citation_contract": citation_contract(),
        "routing_rate": sum(r["ok"] for r in routing) / len(routing),
        "routing": routing,
        "agentic_contract": agentic,
    }
    report["passed"] = (
        retrieval["recall@5"] >= 0.9
        and all(report["citation_contract"].values())
        and report["routing_rate"] >= 0.8
        and all(a["ok"] for a in agentic)
    )
    return report


# ------------------------------------------------------------------------------ live
def run_live(args) -> dict:
    import mlflow
    from mlflow.genai.scorers import Correctness, RelevanceToQuery, RetrievalGroundedness, Safety, scorer

    from retail_insights.agent.factory import Runtime
    from retail_insights.obs.tracing import setup_tracing

    setup_tracing(
        args.experiment,
        *(
            (args.catalog, args.schema, args.warehouse_id)
            if args.trace_to_uc == "true"
            else (None, None, None)
        ),
    )
    rt = Runtime(args.catalog, args.schema, args.warehouse_id, args.endpoint, "vector")
    golden = load_rag_golden()
    has_filings = (
        bool(rt.executor.run(f"SELECT COUNT(*) FROM {rt.layout.filings_meta}").rows[0][0])
        if rt.executor
        else False
    )
    items = [g for g in golden if has_filings or not g.get("requires_filings")]

    retrieval = retrieval_metrics(rt.retriever, [g for g in items if g.get("kind") != "agentic"], k=5)
    rows = []

    def answer(question: str, mode: str) -> str:
        if mode == "agentic":
            r = rt.agentic.run(question)
            rows.append(
                {
                    "question": question,
                    "mode": mode,
                    "grounded": r.grounded,
                    "routes": r.routes,
                    "refused": r.answer == REFUSAL,
                }
            )
        else:
            r = rt.rag.run(question)
            rows.append(
                {
                    "question": question,
                    "mode": mode,
                    "grounded": r.grounded,
                    "routes": ["policies"],
                    "refused": r.answer == REFUSAL,
                }
            )
        return r.answer

    @scorer
    def has_valid_citation(outputs: str) -> bool:
        return bool(CITATION.search(outputs or "")) or outputs == REFUSAL

    def data(subset):
        out = []
        for g in subset:
            rec = {"inputs": {"question": g["question"], "mode": g.get("kind", "rag")}}
            if g["expected_facts"]:
                rec["expectations"] = {"expected_facts": g["expected_facts"]}
            out.append(rec)
        return out

    base = [RelevanceToQuery(), Safety(), RetrievalGroundedness(), has_valid_citation]
    with_facts = [g for g in items if g["expected_facts"]]
    without = [g for g in items if not g["expected_facts"]]
    metrics = {}
    if with_facts:
        metrics.update(
            mlflow.genai.evaluate(
                data=data(with_facts), predict_fn=answer, scorers=[Correctness(), *base]
            ).metrics
        )
    if without:
        res = mlflow.genai.evaluate(data=data(without), predict_fn=answer, scorers=base).metrics
        metrics.update({f"nofacts/{k}": v for k, v in res.items()})

    by_q = {r["question"]: r for r in rows}
    scored = []
    for g in items:
        r = by_q.get(g["question"], {})
        scored.append(
            {
                "id": g["id"],
                "grounded_ok": r.get("refused") if g.get("expect_refusal") else r.get("grounded"),
                "route_ok": route_match(g["routes"], r.get("routes", [])),
            }
        )
    n = len(scored) or 1
    report = {
        "mode": "live",
        "retrieval": retrieval,
        "grounded_rate": sum(bool(s["grounded_ok"]) for s in scored) / n,
        "route_rate": sum(s["route_ok"] for s in scored) / n,
        "judges": metrics,
        "per_question": scored,
    }
    report["passed"] = (
        retrieval["recall@5"] >= args.min_recall
        and report["grounded_rate"] >= args.min_grounded
        and report["route_rate"] >= 0.7
    )
    return report


# ------------------------------------------------------------------------------ main
def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Evaluate RAG and agentic RAG")
    p.add_argument("--offline", action="store_true")
    p.add_argument("--catalog", default="workspace")
    p.add_argument("--schema", default="retail_dev")
    p.add_argument("--warehouse-id", default="")
    p.add_argument("--endpoint", default="databricks-meta-llama-3-3-70b-instruct")
    p.add_argument("--experiment", default="/Shared/retail-ai-platform/evals")
    p.add_argument("--trace-to-uc", default="true")
    p.add_argument("--min-recall", type=float, default=0.8)
    p.add_argument("--min-grounded", type=float, default=0.8)
    p.add_argument("--report", default="")
    args, _ = p.parse_known_args(argv)

    report = run_offline() if args.offline else run_live(args)
    r = report["retrieval"]
    print(f"retrieval recall@5 {r['recall@5']:.0%} | MRR {r['mrr']:.2f}")
    for q in r["per_question"]:
        print(f"  {'HIT ' if q['hit'] else 'MISS'} {q['id']:<26} rank={q['rank']} top={q['top'][:3]}")
    if args.offline:
        print(f"citation contract: {report['citation_contract']}")
        print(f"routing (fallback planner): {report['routing_rate']:.0%}")
        for a in report["agentic_contract"]:
            print(f"  agentic {'PASS' if a['ok'] else 'FAIL'} {a['id']} routes={a['routes']}")
    else:
        print(f"grounded {report['grounded_rate']:.0%} | routes {report['route_rate']:.0%}")
        for k, v in sorted(report["judges"].items()):
            print(f"  judge {k}: {v}")
    if args.report:
        with open(args.report, "w") as f:
            json.dump(report, f, indent=2, default=str)
    if not report["passed"]:
        sys.exit("RAG eval below threshold")


if __name__ == "__main__":
    main()
