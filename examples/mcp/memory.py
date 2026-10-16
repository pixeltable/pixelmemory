# pip install pixelmemory 'mcp>=2.3'
import logging
from datetime import datetime
from functools import lru_cache
import json
import uuid
from typing import Optional

from mcp.server.mcpserver import MCPServer
from pixelmemory import Memory

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
@lru_cache(maxsize=1)
def get_task_memory() -> Memory:
    return Memory(
        context=context,
        namespace="agentic_developer_memory",
        table_name="tasks",
    )


mcp = MCPServer("Pixelmemory tasks")


# --- Agentic Tool for Task Management ---
@mcp.tool()
def manage_tasks(
    action: str,
    task_description: Optional[str] = None,
    task_id: Optional[str] = None,
    status: Optional[str] = None,
) -> str:
    """Manages a persistent to-do list for the development project."""
    task_mem = get_task_memory()
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
            updated_at=str(now),
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

        task_mem.update(
            {"status": status, "updated_at": str(now)},
            where=task_mem.task_id == task_id,
        )

        return f"Task '{task_id}' updated to status '{status}'."

    elif action == "complete":
        if not task_id:
            return "Error: task_id is required for 'complete' action."

        original_task = task_mem.where(task_mem.task_id == task_id).collect()
        if not original_task:
            return f"Error: Task with ID '{task_id}' not found."

        task_mem.update(
            {"status": "completed", "updated_at": str(now)},
            where=task_mem.task_id == task_id,
        )

        return f"Task '{task_id}' marked as completed."

    else:
        return f"Error: Unknown action '{action}'. Valid actions are 'add', 'list', 'update', 'complete'."


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    mcp.run(transport="streamable-http", host="127.0.0.1", port=8000)
