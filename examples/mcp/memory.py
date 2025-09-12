import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator
from datetime import datetime
import json
import uuid
from typing import Optional

import pixeltable as pxt
import uvicorn
from mcp.server.fastmcp import FastMCP
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from pixelmemory import Memory
from starlette.applications import Starlette
from starlette.routing import Mount
from starlette.types import Receive, Scope, Send

# --- Memory Context ---
from pixelmemory.context import Text

context = [
    Text(id="task_id", embed=False),
    Text(id="description", embed=True),
    Text(id="status", embed=False),
    Text(id="created_at", embed=False),
    Text(id="updated_at", embed=False),
]

# --- Memory Initialization ---
task_mem = Memory(
    context=context,
    namespace="agentic_developer_memory",
    table_name="tasks",
    if_exists="ignore"
)

mcp = FastMCP()


# --- Agentic Tool for Task Management ---
@mcp.tool()
def manage_tasks(
    action: str,
    task_description: Optional[str] = None,
    task_id: Optional[str] = None,
    status: Optional[str] = None,
) -> str:
    """Manages a persistent to-do list for the development project."""
    now = datetime.now()

    if action == "add":
        if not task_description:
            return "Error: task_description is required for 'add' action."
        new_task_id = str(uuid.uuid4())
        entry = task_mem.Entry(
            task_id=new_task_id,
            description=task_description,
            status="pending",
            created_at=str(now),
            updated_at=str(now)
        )
        task_mem.add(entry)
        return f"Task added with ID: {new_task_id}"

    elif action == "list":
        tasks = (
            task_mem.select(
                task_mem.task_id,
                task_mem.description,
                task_mem.status,
                task_mem.updated_at,
            )
            .order_by(task_mem.updated_at, asc=False)
            .collect()
        )

        if not tasks:
            return "No tasks found."

        task_list = [{k: v for k, v in task.items()} for task in tasks]
        return json.dumps(task_list, indent=2, default=str)

    elif action == "update":
        if not task_id or not status:
            return "Error: task_id and status are required for 'update' action."

        original_task = task_mem.where(task_mem.task_id == task_id).collect()
        if not original_task:
            return f"Error: Task with ID '{task_id}' not found."

        task_mem.delete(task_mem.task_id == task_id)

        original = original_task[0]
        updated_entry = task_mem.Entry(
            task_id=original["task_id"],
            description=original["description"],
            status=status,
            created_at=str(original["created_at"]),
            updated_at=str(now)
        )
        task_mem.add(updated_entry)

        return f"Task '{task_id}' updated to status '{status}'."

    elif action == "complete":
        if not task_id:
            return "Error: task_id is required for 'complete' action."

        original_task = task_mem.where(task_mem.task_id == task_id).collect()
        if not original_task:
            return f"Error: Task with ID '{task_id}' not found."

        task_mem.delete(task_mem.task_id == task_id)

        original = original_task[0]
        completed_entry = task_mem.Entry(
            task_id=original["task_id"],
            description=original["description"],
            status="completed",
            created_at=str(original["created_at"]),
            updated_at=str(now)
        )
        task_mem.add(completed_entry)

        return f"Task '{task_id}' marked as completed."

    else:
        return f"Error: Unknown action '{action}'. Valid actions are 'add', 'list', 'update', 'complete'."


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    session_manager = StreamableHTTPSessionManager(app=mcp)

    async def handle_streamable_http(
        scope: Scope, receive: Receive, send: Send
    ) -> None:
        await session_manager.handle_request(scope, receive, send)

    @contextlib.asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        """Context manager for managing session manager lifecycle."""
        async with session_manager.run():
            logger.info("Application started with StreamableHTTP session manager!")
            try:
                yield
            finally:
                logger.info("Application shutting down...")

    starlette_app = Starlette(
        debug=True,
        routes=[
            Mount("/", app=handle_streamable_http),
        ],
        lifespan=lifespan,
    )

    uvicorn.run(starlette_app, host="127.0.0.1", port=8000)
