"""
Memory management system for AI applications.
"""

import pixeltable as pxt
from pixeltable.functions.huggingface import sentence_transformer
from typing import List, Dict, Any
import json
import datetime

class Memory:
    """A class for managing and retrieving AI conversation memories."""
    
    def __init__(self, embedding_model: str = "intfloat/e5-large-v2"):
        """
        Initialize the Memory system.
        
        Args:
            embedding_model: The embedding model to use for semantic search
        """
        self.embedding_model = sentence_transformer.using(model_id=embedding_model)
        
    def _get_or_create_memory_table(self, user_id: str) -> pxt.Table:
        """Get or create a memory table for the specified user."""
        table_name = f"pixelmemory_{user_id}"
        
        try:
            table = pxt.get_table(table_name)
        except:
            # Create new table if it doesn't exist
            table = pxt.create_table(
                table_name,
                schema={
                    "timestamp": pxt.Timestamp,
                    "memory": pxt.String,
                    "metadata": pxt.String,
                }
            )
            
            # Add embedding index
            table.add_embedding_index(
                column="memory",
                idx_name="memory_idx",
                string_embed=self.embedding_model,
                if_exists="ignore",
            )
            
        return table
    
    def add(self, messages: List[Dict[str, str]], user_id: str = "default_user") -> None:
        """
        Add conversation messages to memory.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            user_id: Identifier for the user whose memory to update
        """
        table = self._get_or_create_memory_table(user_id)
        
        # Extract the conversation as a memory
        conversation = " ".join([f"{msg['role']}: {msg['content']}" for msg in messages])
        
        # Store the memory
        table.insert({
            "timestamp": datetime.datetime.now(),
            "memory": conversation,
            "metadata": json.dumps({"messages": messages})
        })
    
    def search(self, query: str, user_id: str = "default_user", limit: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        """
        Search for relevant memories based on semantic similarity.
        
        Args:
            query: The search query
            user_id: Identifier for the user whose memories to search
            limit: Maximum number of results to return
            
        Returns:
            Dictionary with search results
        """
        table = self._get_or_create_memory_table(user_id)
        
        # Perform semantic search
        sim = table.memory.similarity(query, idx="memory_idx")
        results = (
            table.order_by(sim, asc=False)
            .select(table.timestamp, table.memory, table.metadata, similarity=sim)
            .limit(limit)
            .collect()
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results["memory"])):
            formatted_results.append({
                "timestamp": results["timestamp"][i].isoformat() if results["timestamp"][i] else None,
                "memory": results["memory"][i],
                "metadata": json.loads(results["metadata"][i]) if results["metadata"][i] else {},
                "similarity": float(results["similarity"][i])
            })
            
        return {"results": formatted_results}
