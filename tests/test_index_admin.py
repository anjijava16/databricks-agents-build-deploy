from types import SimpleNamespace

from databricks.sdk.errors import NotFound

from retail_insights.rag import index_admin


class FakeIndexes:
    def __init__(self, exists: bool, statuses=()):
        self.exists, self.created, self.statuses = exists, None, list(statuses)

    def get_index(self, name):
        if not self.exists:
            raise NotFound("missing")
        st = self.statuses.pop(0) if self.statuses else (True, 99)
        return SimpleNamespace(status=SimpleNamespace(ready=st[0], indexed_row_count=st[1]))

    def create_index(self, **kw):
        self.created = kw


class FakeEndpoints:
    def __init__(self, exists: bool):
        self.exists, self.created = exists, None

    def get_endpoint(self, name):
        if not self.exists:
            raise NotFound("missing")
        return SimpleNamespace(endpoint_status=SimpleNamespace(state=SimpleNamespace(value="ONLINE")))

    def create_endpoint_and_wait(self, name, endpoint_type):
        self.created = (name, endpoint_type.value)


def test_ensure_endpoint_creates_once():
    eps = FakeEndpoints(exists=False)
    index_admin.ensure_endpoint(SimpleNamespace(vector_search_endpoints=eps), "retail-ai-vs")
    assert eps.created == ("retail-ai-vs", "STANDARD")
    eps = FakeEndpoints(exists=True)
    index_admin.ensure_endpoint(SimpleNamespace(vector_search_endpoints=eps), "retail-ai-vs")
    assert eps.created is None


def test_ensure_index_spec():
    idx = FakeIndexes(exists=False)
    ws = SimpleNamespace(vector_search_indexes=idx)
    created = index_admin.ensure_index(
        ws,
        "c.s.gold_doc_chunks_index",
        "retail-ai-vs",
        "c.s.gold_doc_chunks",
        "databricks-gte-large-en",
        ["chunk_id", "content"],
    )
    spec = idx.created["delta_sync_index_spec"]
    assert created and idx.created["primary_key"] == "chunk_id"
    assert spec.source_table == "c.s.gold_doc_chunks" and spec.pipeline_type.value == "TRIGGERED"
    assert spec.embedding_source_columns[0].name == "embed_text"
    assert spec.embedding_source_columns[0].embedding_model_endpoint_name == "databricks-gte-large-en"
    assert not index_admin.ensure_index(
        SimpleNamespace(vector_search_indexes=FakeIndexes(True)), "i", "e", "t", "m", []
    )


def test_wait_until_synced(monkeypatch):
    monkeypatch.setattr(index_admin.time, "sleep", lambda s: None)
    idx = FakeIndexes(exists=True, statuses=[(False, 0), (True, 50), (True, 120)])
    index_admin.wait_until_synced(SimpleNamespace(vector_search_indexes=idx), "i", expected_rows=120)
    assert idx.statuses == []
