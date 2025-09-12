from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from pixelmemory import Memory
from pixelmemory.context import Text, Image, Video, Audio, Document
import uvicorn

app = FastAPI(
    title="AI Memory Service",
    description="A FastAPI service for creating, managing, and searching multimodal memories with context-based API.",
    version="2.0.0",
)

# --- Pydantic Models for Request/Response ---


class ContextField(BaseModel):
    id: str
    type: str  # "text", "image", "video", "audio", "document"
    embed: Optional[bool] = True
    provider: Optional[str] = None  # For image/video
    model: Optional[str] = None     # For image/video

class CreateMemoryRequest(BaseModel):
    namespace: str
    table_name: str
    context: List[ContextField]
    if_exists: Optional[str] = "ignore"


class AddItemsRequest(BaseModel):
    items: List[Dict[str, Any]]


class MemoryInfoResponse(BaseModel):
    namespace: str
    table_name: str
    context: List[Dict[str, Any]]
    metadata: Dict[str, Any]


# A simple cache for initialized Memory objects
memory_cache: Dict[str, Memory] = {}


def get_memory(namespace: str, table_name: str) -> Memory:
    """Helper to get a Memory instance, caching it for efficiency."""
    cache_key = f"{namespace}.{table_name}"
    if cache_key in memory_cache:
        return memory_cache[cache_key]

    # Note: We can't recreate a Memory from scratch without the original context
    # In a real application, you'd store the context configuration alongside the memory
    raise HTTPException(
        status_code=404, 
        detail=f"Memory '{cache_key}' not found in cache. Use create_memory first."
    )


def context_from_fields(context_fields: List[ContextField]) -> List:
    """Converts context field definitions to context objects."""
    context = []
    
    for field in context_fields:
        if field.type == "text":
            context.append(Text(id=field.id, embed=field.embed))
        elif field.type == "image":
            context.append(Image(
                id=field.id, 
                provider=field.provider or "openai",
                model=field.model or "gpt-4o-mini"
            ))
        elif field.type == "video":
            context.append(Video(
                id=field.id,
                provider=field.provider or "openai", 
                model=field.model or "gpt-4o-mini"
            ))
        elif field.type == "audio":
            context.append(Audio(id=field.id))
        elif field.type == "document":
            context.append(Document(id=field.id))
        else:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported context type: {field.type}"
            )
    
    return context


# --- API Endpoints ---


@app.post("/memories", status_code=201)
def create_memory(req: CreateMemoryRequest):
    """
    Creates a new memory table using context-based API.
    """
    try:
        # Convert context fields to context objects
        context = context_from_fields(req.context)

        # Create the memory instance
        memory = Memory(
            context=context,
            namespace=req.namespace,
            table_name=req.table_name,
            if_exists=req.if_exists
        )
        
        # Cache the memory instance
        cache_key = f"{req.namespace}.{req.table_name}"
        memory_cache[cache_key] = memory
        
        return {
            "message": f"Memory '{req.namespace}.{req.table_name}' created successfully."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create memory: {e}")


@app.post("/memories/{namespace}/{table_name}/items", status_code=201)
def add_items(namespace: str, table_name: str, req: AddItemsRequest):
    """
    Adds one or more items to a memory table using Entry API.
    """
    mem = get_memory(namespace, table_name)
    try:
        # Convert items to Entry objects
        entries = []
        for item in req.items:
            entry = mem.Entry(**item)
            entries.append(entry)
        
        mem.add(*entries)
        return {"message": f"Successfully added {len(req.items)} items."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add items: {e}")


@app.get("/memories/{namespace}/{table_name}/items")
def search_items(
    namespace: str,
    table_name: str,
    query: Optional[str] = None,
    search_column: Optional[str] = None,
    filter_expression: Optional[str] = Query(None, alias="filter"),
    limit: int = 10,
    select: Optional[str] = None,
):
    """
    Searches and retrieves items from a memory.
    """
    mem = get_memory(namespace, table_name)

    try:
        q = mem

        # Apply semantic search if query and search_column are provided
        if query and search_column:
            try:
                similarity = getattr(mem, search_column).similarity(query)
                q = q.order_by(similarity, asc=False)
            except AttributeError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Search column '{search_column}' not found or not searchable.",
                )

        # Apply filter expression
        if filter_expression:
            # Note: Direct filter expression parsing would require more complex implementation
            # For now, we skip this feature in the context-based API
            raise HTTPException(
                status_code=400,
                detail="Filter expressions not yet supported in context-based API"
            )

        # Apply select columns
        if select:
            select_cols = [col.strip() for col in select.split(",")]
            q = q.select(*select_cols)

        # Apply limit
        results = q.limit(limit).collect()

        # Convert results to a list of dicts for JSON response
        return [{k: v for k, v in row.items()} for row in results]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to search items: {e}")


@app.get("/memories", response_model=List[str])
def list_memories(namespace: Optional[str] = None):
    """
    Lists all available memory tables, optionally filtered by namespace.
    """
    try:
        # This is a simplified way to list tables.
        # In a real-world scenario, you might have a more robust way of tracking tables.
        # Here, we list directories in the .pixeltable path.
        from pathlib import Path
        import os

        pxt_dir = Path(os.environ.get("PIXELTABLE_HOME", Path.home() / ".pixeltable"))
        if not pxt_dir.exists():
            return []

        tables = []
        if namespace:
            ns_path = pxt_dir / namespace
            if ns_path.exists():
                tables = [
                    f"{namespace}.{d.name}" for d in ns_path.iterdir() if d.is_dir()
                ]
        else:
            for ns_dir in pxt_dir.iterdir():
                if ns_dir.is_dir():
                    for table_dir in ns_dir.iterdir():
                        if table_dir.is_dir():
                            tables.append(f"{ns_dir.name}.{table_dir.name}")
        return tables
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list memories: {e}")


@app.get("/memories/{namespace}/{table_name}", response_model=MemoryInfoResponse)
def get_memory_info(namespace: str, table_name: str):
    """
    Retrieves metadata and context information for a specific memory table.
    """
    mem = get_memory(namespace, table_name)
    try:
        metadata = mem.table.get_metadata()
        
        # Extract context information from the memory instance
        context_info = []
        for ctx in mem.context:
            ctx_dict = {
                "id": ctx.id,
                "type": ctx.__class__.__name__.lower(),
                "embed": ctx.embed
            }
            if hasattr(ctx, 'provider'):
                ctx_dict["provider"] = ctx.provider
            if hasattr(ctx, 'model'):
                ctx_dict["model"] = ctx.model
            context_info.append(ctx_dict)
        
        return {
            "namespace": namespace,
            "table_name": table_name,
            "context": context_info,
            "metadata": metadata,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get memory info: {e}")


@app.delete("/memories/{namespace}/{table_name}", status_code=200)
def delete_memory(namespace: str, table_name: str):
    """
    Deletes a memory table.
    """
    mem = get_memory(namespace, table_name)
    try:
        mem.drop()
        # Remove from cache if it exists
        cache_key = f"{namespace}.{table_name}"
        if cache_key in memory_cache:
            del memory_cache[cache_key]
        return {"message": f"Memory '{namespace}.{table_name}' deleted successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete memory: {e}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
