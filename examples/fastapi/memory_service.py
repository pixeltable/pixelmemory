from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from pixelmemory import Memory
import pixeltable as pxt
import uvicorn

app = FastAPI(
    title="AI Memory Service",
    description="A FastAPI service for creating, managing, and searching multimodal memories.",
    version="1.0.0",
)

# --- Pydantic Models for Request/Response ---

class CreateMemoryRequest(BaseModel):
    namespace: str
    table_name: str
    schema: Dict[str, str]
    columns_to_index: List[str]
    if_exists: Optional[str] = "ignore"
    primary_key: Optional[str] = None

class AddItemsRequest(BaseModel):
    items: List[Dict[str, Any]]

class MemoryInfoResponse(BaseModel):
    namespace: str
    table_name: str
    schema: Dict[str, Any]
    columns_to_index: List[str]
    metadata: Dict[str, Any]

# A simple cache for initialized Memory objects
memory_cache: Dict[str, Memory] = {}

def get_memory(namespace: str, table_name: str) -> Memory:
    """Helper to get a Memory instance, caching it for efficiency."""
    cache_key = f"{namespace}.{table_name}"
    if cache_key in memory_cache:
        return memory_cache[cache_key]
    
    try:
        mem = Memory(namespace=namespace, table_name=table_name)
        memory_cache[cache_key] = mem
        return mem
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Memory '{cache_key}' not found: {e}")

def pxt_type_from_string(type_str: str) -> pxt.ColumnType:
    """Converts a string to a pixeltable type."""
    type_map = {
        "string": pxt.String,
        "int": pxt.Int,
        "float": pxt.Float,
        "bool": pxt.Bool,
        "timestamp": pxt.Timestamp,
        "json": pxt.Json,
        "image": pxt.Image,
        "video": pxt.Video,
        "audio": pxt.Audio,
        "document": pxt.Document,
    }
    if type_str.lower() in type_map:
        return type_map[type_str.lower()]()
    raise HTTPException(status_code=400, detail=f"Unsupported pxt type: {type_str}")


# --- API Endpoints ---

@app.post("/memories", status_code=201)
def create_memory(req: CreateMemoryRequest):
    """
    Creates a new memory table.
    """
    try:
        # Convert string schema to pixeltable types
        pxt_schema = {col: pxt_type_from_string(type_str) for col, type_str in req.schema.items()}
        
        Memory(
            namespace=req.namespace,
            table_name=req.table_name,
            schema=pxt_schema,
            columns_to_index=req.columns_to_index,
            if_exists=req.if_exists,
            primary_key=req.primary_key
        )
        return {"message": f"Memory '{req.namespace}.{req.table_name}' created successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create memory: {e}")


@app.post("/memories/{namespace}/{table_name}/items", status_code=201)
def add_items(namespace: str, table_name: str, req: AddItemsRequest):
    """
    Adds one or more items to a memory table.
    """
    mem = get_memory(namespace, table_name)
    try:
        mem.insert(req.items)
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
            if search_column not in mem.schema:
                raise HTTPException(status_code=400, detail=f"Search column '{search_column}' not in schema.")
            similarity = getattr(mem, search_column).similarity(query)
            q = q.order_by(similarity, asc=False)
        
        # Apply filter expression
        if filter_expression:
            q = q.where(pxt.expr(filter_expression))
            
        # Apply select columns
        if select:
            select_cols = [col.strip() for col in select.split(',')]
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
                tables = [f"{namespace}.{d.name}" for d in ns_path.iterdir() if d.is_dir()]
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
    Retrieves metadata and schema information for a specific memory table.
    """
    mem = get_memory(namespace, table_name)
    try:
        metadata = mem.table.get_metadata()
        # The schema from get_metadata is already in a serializable format
        return {
            "namespace": namespace,
            "table_name": table_name,
            "schema": metadata['schema'],
            "columns_to_index": [col['name'] for col in metadata['cols'] if col.get('is_indexed')],
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
