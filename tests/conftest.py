"""Keyless, offline test environment.

Two things this file guarantees, and the whole suite depends on both:

1. Every run gets its own throwaway Pixeltable catalog, so tests never see or
   corrupt a developer's `~/.pixeltable`.
2. `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` are removed from the environment
   before Pixeltable is imported. A test that needs a key fails here rather
   than passing on the machine that happens to have one.

Both must happen before `import pixeltable`, which is why they run at module
import time and why no test module imports pixelmemory at the top level.
"""

import os
import tempfile

_CATALOG = tempfile.mkdtemp(prefix="pixelmemory-tests-")
os.environ["PIXELTABLE_HOME"] = _CATALOG
os.environ.pop("OPENAI_API_KEY", None)
os.environ.pop("ANTHROPIC_API_KEY", None)

import pytest  # noqa: E402

# A small model, and one that is never actually loaded: the tests construct
# schemas and never insert, so no weights are downloaded.
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@pytest.fixture(scope="session")
def embed_model() -> str:
    return EMBED_MODEL


@pytest.fixture
def namespace(request: pytest.FixtureRequest) -> str:
    """A namespace unique to the test, so tests cannot collide in the catalog.

    Lowercased because Pixeltable normalizes catalog paths to lower case.
    """
    name = request.node.name.replace("[", "_").replace("]", "").replace("-", "_")
    return ("t_" + name).lower()
