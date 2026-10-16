"""Exercise example entry points without contacting a model provider."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace

import numpy as np
import pixeltable as pxt
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, messages_from_dict

from pixelmemory import Memory
from pixelmemory.context import Text


def load_example(relative_path):
    path = Path(__file__).resolve().parent.parent / "examples" / relative_path
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ChatModel:
    def __init__(self):
        self.calls = []

    def invoke(self, messages):
        self.calls.append(list(messages))
        return AIMessage(content=f"answer {len(self.calls)}")


def test_chat_history_survives_turns_and_reconstruction(namespace):
    example = load_example("integrations/langchain_chat_history.py")
    memory = Memory(example.context, namespace=namespace)
    model = ChatModel()
    for text in ("first question", "second question", "third question"):
        example.chat(memory, model, "session_a", text)

    assert [m.content for m in model.calls[2]][1:] == [
        "first question",
        "answer 1",
        "second question",
        "answer 2",
        "third question",
    ]
    assert memory.count() == 1
    history = messages_from_dict(json.loads(memory.collect()[0]["messages"]))
    assert len(history) == 7

    reopened = Memory(example.context, namespace=namespace)
    example.chat(reopened, model, "session_a", "after reopening")
    assert len(model.calls[-1]) == 8
    example.chat(reopened, model, "session_b", "separate session")
    assert len(model.calls[-1]) == 2
    assert reopened.count() == 2


@pxt.udf
def example_embedding(text: str) -> pxt.Array[(3,), pxt.Float]:
    """A local embedding keeps example tests independent of downloaded weights."""
    return np.array([1.0, float("Nvidia" in text), 1.0])


def test_financial_chat_preserves_history_and_filters_metadata(namespace):
    example = load_example("integrations/langchain_financial_rag_agent.py")
    memory = Memory(example.chat_context, namespace=namespace)
    chunks = pxt.create_table(
        f"{namespace}.report_chunks", {"text": pxt.String, "company": pxt.String}
    )
    chunks.add_embedding_index(
        "text", embedding=example_embedding, idx_name="report_idx"
    )
    chunks.insert(
        [
            {"text": "Nvidia report excerpt", "company": "Nvidia"},
            {"text": "AMD report excerpt", "company": "AMD"},
        ]
    )
    # Use a real indexed table for retrieval, without fetching a remote PDF.
    kb_memory = SimpleNamespace(
        views=lambda name: {name: chunks},
        search_targets=[SimpleNamespace(context_id="report", index_name="report_idx")],
    )
    model = ChatModel()
    for text in ("first question", "second question", "third question"):
        example.financial_chat(
            kb_memory, memory, model, "session_a", text, {"company": "Nvidia"}
        )

    third_turn = model.calls[-1]
    assert len(third_turn) == 6
    assert [m.type for m in third_turn] == [
        "system",
        "human",
        "ai",
        "human",
        "ai",
        "human",
    ]
    assert "Nvidia report excerpt" in third_turn[-1].content[0]["text"]
    assert "AMD report excerpt" not in third_turn[-1].content[0]["text"]
    assert memory.count() == 1
    assert len(messages_from_dict(json.loads(memory.collect()[0]["messages"]))) == 7


def test_fastapi_memory_lifecycle_and_client_errors(namespace):
    example = load_example("fastapi/memory_service.py")
    with TestClient(example.app) as client:
        payload = {
            "namespace": namespace,
            "table_name": "entries",
            "context": [{"id": "body", "type": "text", "embed": False}],
        }
        response = client.post("/memories", json=payload)
        assert response.status_code == 201, response.text
        assert client.get("/memories").json() == [f"{namespace}.entries"]
        assert client.get("/memories", params={"namespace": "other"}).json() == []

        path = f"/memories/{namespace}/entries"
        assert (
            client.post(
                f"{path}/items", json={"items": [{"body": "hello"}]}
            ).status_code
            == 201
        )
        assert client.get(f"{path}/items", params={"select": "body"}).json() == [
            {"body": "hello"}
        ]
        assert client.get(path).status_code == 200
        for params in (
            {"filter": "body = hello"},
            {"select": "missing"},
            {"query": "hello"},
            {"query": "hello", "search_column": "missing"},
        ):
            response = client.get(f"{path}/items", params=params)
            assert response.status_code == 400, response.text
        assert client.get(f"{path}/items", params={"limit": 0}).status_code == 422
        assert (
            client.post(
                f"{path}/items", json={"items": [{"unknown": "value"}]}
            ).status_code
            == 400
        )

        invalid = {**payload, "context": [{"id": "body", "type": "unknown"}]}
        assert client.post("/memories", json=invalid).status_code == 400
        assert client.delete(path).status_code == 200
        assert client.get("/memories").json() == []
        assert client.get(path).status_code == 404
        assert not pxt.list_tables(namespace)


def test_fastapi_semantic_search_uses_registered_targets(namespace):
    example = load_example("fastapi/memory_service.py")
    memory = Memory(
        [Text(id="body", embed_model=example_embedding)],
        namespace=namespace,
    )
    memory.add(memory.Entry(body="Nvidia report"))
    example.memory_cache[f"{namespace}.memory"] = memory
    with TestClient(example.app) as client:
        response = client.get(
            f"/memories/{namespace}/memory/items",
            params={"query": "Nvidia", "search_column": "body"},
        )
        assert response.status_code == 200, response.text
        assert response.json()[0]["text"] == "Nvidia report"
        assert response.json()[0]["context_id"] == "body"


@pytest.mark.parametrize("kind", ["text", "image", "video", "audio", "document"])
def test_fastapi_honors_embed_false_for_every_context(kind):
    example = load_example("fastapi/memory_service.py")
    context = example.context_from_fields(
        [example.ContextField(id="content", type=kind, embed=False)]
    )
    assert context[0].embed is False


def test_mcp_http_initialization_and_task_lifecycle(namespace, monkeypatch):
    example = load_example("mcp/memory.py")
    # Importing or connecting to the server must not initialize a catalog or
    # fetch embeddings; the memory is first needed when a tool runs.
    assert example.get_task_memory.cache_info().currsize == 0
    app = example.mcp.streamable_http_app(stateless_http=True, json_response=True)
    with TestClient(app, base_url="http://127.0.0.1:8000") as client:
        client.headers["Accept"] = "application/json, text/event-stream"

        def rpc(method, params):
            response = client.post(
                "/mcp",
                json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
            )
            assert response.status_code == 200, response.text
            assert "error" not in response.json(), response.text
            return response.json()["result"]

        initialized = rpc(
            "initialize",
            {
                "protocolVersion": "2025-11-25",
                "capabilities": {},
                "clientInfo": {"name": "pixelmemory-test", "version": "0.1"},
            },
        )
        assert initialized["serverInfo"]["name"] == "Pixelmemory tasks"
        assert example.get_task_memory.cache_info().currsize == 0

        memory = Memory(
            [replace(ctx, embed=False) for ctx in example.context], namespace=namespace
        )
        monkeypatch.setattr(example, "get_task_memory", lambda: memory)
        assert rpc("tools/list", {})["tools"][0]["name"] == "manage_tasks"

        def call(**arguments):
            result = rpc("tools/call", {"name": "manage_tasks", "arguments": arguments})
            assert not result.get("isError"), result
            return result["content"][0]["text"]

        assert call(action="list") == "No tasks found."
        call(action="add", task_description="Keep my task")
        task = memory.collect()[0]
        task_id = task["task_id"]
        call(action="update", task_id=task_id, status="in_progress")
        call(action="complete", task_id=task_id)
        assert memory.count() == 1
        updated = memory.collect()[0]
        assert updated["status"] == "completed"
        assert updated["description"] == task["description"]
        assert updated["created_at"] == task["created_at"]
        assert json.loads(call(action="list"))[0]["status"] == "completed"
        assert "not found" in call(action="complete", task_id="missing")
