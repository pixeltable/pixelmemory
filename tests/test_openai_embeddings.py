"""
Tests for OpenAI embedding integration with PixelMemory.
"""

import pytest
import pixeltable as pxt
from pixelmemory import Memory
from pixeltable.functions.openai import embeddings

def test_openai_embeddings_basic():
    """Test basic OpenAI embeddings functionality."""
    # Clean up any existing test data
    try:
        pxt.drop_dir('test_openai_embeddings', force=True)
    except:
        pass
    
    # Create Memory with OpenAI embeddings
    memory = Memory(
        namespace="test_openai_embeddings",
        table_name="test_docs",
        schema={
            'doc_id': pxt.Required[pxt.String],
            'content': pxt.String,
        },
        columns_to_index=['content'],
        text_embedding_model=embeddings.using(model="text-embedding-3-small"),
        primary_key='doc_id'
    )
    
    # Insert test documents
    test_docs = [
        {
            'doc_id': 'doc1',
            'content': 'Machine learning and artificial intelligence applications'
        },
        {
            'doc_id': 'doc2',
            'content': 'Python programming for data science projects'
        }
    ]
    
    memory.insert(test_docs)
    
    # Test similarity search
    query = "AI and ML applications"
    similarity = memory.content.similarity(query)
    
    results = (memory
               .order_by(similarity, asc=False)
               .select(memory.doc_id, memory.content, score=similarity)
               .limit(2)
               .collect())
    
    # Verify results
    assert len(results) == 2
    assert results[0]['doc_id'] in ['doc1', 'doc2']
    assert 'score' in results[0]
    assert isinstance(results[0]['score'], (int, float))
    
    # Clean up
    pxt.drop_dir('test_openai_embeddings', force=True)

def test_openai_embeddings_custom_dimensions():
    """Test OpenAI embeddings with custom dimensions."""
    try:
        pxt.drop_dir('test_openai_custom_dims', force=True)
    except:
        pass
    
    # Create Memory with custom dimensions
    memory = Memory(
        namespace="test_openai_custom_dims",
        table_name="test_docs",
        schema={
            'doc_id': pxt.Required[pxt.String],
            'content': pxt.String,
        },
        columns_to_index=['content'],
        text_embedding_model=embeddings.using(
            model="text-embedding-3-small",
            dimensions=512  # Custom dimension size
        ),
        primary_key='doc_id'
    )
    
    # Insert and test
    test_doc = [{'doc_id': 'test1', 'content': 'Test content for custom dimensions'}]
    memory.insert(test_doc)
    
    # Verify the table exists and has content
    results = memory.select(memory.doc_id, memory.content).collect()
    assert len(results) == 1
    assert results[0]['doc_id'] == 'test1'
    
    # Clean up
    pxt.drop_dir('test_openai_custom_dims', force=True)

def test_openai_embeddings_comparison():
    """Test that we can create Memory instances with different OpenAI models."""
    try:
        pxt.drop_dir('test_openai_comparison', force=True)
    except:
        pass
    
    # Test different OpenAI models
    models_to_test = [
        "text-embedding-3-small",
        "text-embedding-ada-002"
    ]
    
    for model_name in models_to_test:
        # Create memory with specific model
        memory = Memory(
            namespace="test_openai_comparison",
            table_name=f"test_{model_name.replace('-', '_')}",
            schema={'content': pxt.String},
            columns_to_index=['content'],
            text_embedding_model=embeddings.using(model=model_name),
            if_exists="replace_force"
        )
        
        # Insert test data
        memory.insert([{'content': f'Test content for {model_name}'}])
        
        # Verify data exists
        results = memory.select(memory.content).collect()
        assert len(results) == 1
        assert model_name in results[0]['content']
    
    # Clean up
    pxt.drop_dir('test_openai_comparison', force=True)

if __name__ == "__main__":
    test_openai_embeddings_basic()
    test_openai_embeddings_custom_dimensions()
    test_openai_embeddings_comparison()
    print("All OpenAI embedding tests passed!")
