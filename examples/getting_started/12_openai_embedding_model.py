"""
Example demonstrating OpenAI embedding models with PixelMemory.

This example shows how to:
1. Use OpenAI embedding models instead of Hugging Face models
2. Configure different OpenAI embedding models
3. Compare embedding performance across models
4. Perform semantic search with OpenAI embeddings
"""

from pixelmemory import Memory
import pixeltable as pxt
from pixeltable.functions.openai import embeddings

# Example 1: Basic OpenAI embeddings usage
print("=== Example 1: Basic OpenAI Embeddings ===")

schema = {
    "doc_id": pxt.Required[pxt.String], 
    "content": pxt.String,
    "category": pxt.String
}

# Create Memory instance with OpenAI embedding model
memory_openai = Memory(
    namespace="openai_embeddings_demo",
    table_name="documents",
    schema=schema,
    columns_to_index=["content"],
    text_embedding_model=embeddings.using(model="text-embedding-3-small"),
    primary_key="doc_id",
    if_exists="replace_force"
)

# Sample documents for testing
sample_docs = [
    {
        "doc_id": "tech_001",
        "content": "Machine learning algorithms enable computers to learn from data without explicit programming. Neural networks are a subset of machine learning inspired by biological neural networks.",
        "category": "technology"
    },
    {
        "doc_id": "tech_002", 
        "content": "Python is a high-level programming language known for its simplicity and readability. It's widely used in data science, web development, and artificial intelligence.",
        "category": "technology"
    },
    {
        "doc_id": "science_001",
        "content": "Photosynthesis is the process by which plants convert sunlight, carbon dioxide, and water into glucose and oxygen. This process is fundamental to life on Earth.",
        "category": "science"
    },
    {
        "doc_id": "science_002",
        "content": "The theory of relativity, proposed by Einstein, revolutionized our understanding of space, time, and gravity. It consists of special and general relativity.",
        "category": "science"
    }
]

print("Inserting sample documents...")
memory_openai.insert(sample_docs)

# Perform semantic search
query = "artificial intelligence and programming"
print(f"\nSearching for: '{query}'")

similarity = memory_openai.content.similarity(query)
results = (
    memory_openai.order_by(similarity, asc=False)
    .select(
        memory_openai.doc_id,
        memory_openai.content,
        memory_openai.category,
        score=similarity
    )
    .limit(3)
    .collect()
)

print("Search results:")
for result in results:
    print(f"ID: {result['doc_id']}")
    print(f"Category: {result['category']}")
    print(f"Score: {result['score']:.4f}")
    print(f"Content: {result['content'][:100]}...")
    print()

# Example 2: Compare different OpenAI embedding models
print("=== Example 2: Comparing OpenAI Embedding Models ===")

# Test different OpenAI embedding models
embedding_models = [
    ("text-embedding-3-small", 1536),   # Smaller, faster, cheaper
    ("text-embedding-3-large", 3072),   # Larger, more accurate, more expensive
    ("text-embedding-ada-002", 1536),   # Legacy model
]

comparison_results = {}

for model_name, dimensions in embedding_models:
    print(f"\nTesting model: {model_name} ({dimensions} dimensions)")
    
    try:
        # Create memory with specific OpenAI model
        memory_test = Memory(
            namespace="openai_embeddings_demo",
            table_name=f"test_{model_name.replace('-', '_')}",
            schema={"text": pxt.String},
            columns_to_index=["text"],
            text_embedding_model=embeddings.using(model=model_name),
            if_exists="replace_force"
        )
        
        # Insert test documents
        test_docs = [
            {"text": "Python programming and machine learning"},
            {"text": "Data science with artificial intelligence"},
            {"text": "Web development using modern frameworks"}
        ]
        memory_test.insert(test_docs)
        
        # Test search
        test_query = "AI and programming"
        sim = memory_test.text.similarity(test_query)
        search_results = (
            memory_test.order_by(sim, asc=False)
            .select(memory_test.text, score=sim)
            .limit(1)
            .collect()
        )
        
        if search_results:
            comparison_results[model_name] = {
                "dimensions": dimensions,
                "top_score": search_results[0]['score'],
                "top_match": search_results[0]['text']
            }
            print(f"✓ Top match score: {search_results[0]['score']:.4f}")
            print(f"  Best match: {search_results[0]['text']}")
        
    except Exception as e:
        print(f"✗ Error with {model_name}: {str(e)}")
        comparison_results[model_name] = {"error": str(e)}

# Display comparison summary
print("\n=== Model Comparison Summary ===")
for model, results in comparison_results.items():
    if "error" not in results:
        print(f"{model}: {results['dimensions']} dims, score: {results['top_score']:.4f}")
    else:
        print(f"{model}: Error - {results['error']}")

# Example 3: Advanced usage with custom dimensions
print("\n=== Example 3: Custom Dimensions with OpenAI Embeddings ===")

try:
    # text-embedding-3-small and text-embedding-3-large support custom dimensions
    custom_memory = Memory(
        namespace="openai_embeddings_demo", 
        table_name="custom_dimensions",
        schema={"content": pxt.String},
        columns_to_index=["content"],
        text_embedding_model=embeddings.using(
            model="text-embedding-3-small",
            dimensions=512  # Reduce from default 1536 to 512
        ),
        if_exists="replace_force"
    )
    
    # Insert test content
    custom_docs = [
        {"content": "Reduced dimensionality embeddings for efficiency"},
        {"content": "Custom embedding dimensions with OpenAI models"},
        {"content": "Optimizing embedding size for storage and performance"}
    ]
    custom_memory.insert(custom_docs)
    
    # Test search with custom dimensions
    query = "embedding optimization"
    sim = custom_memory.content.similarity(query) 
    results = (
        custom_memory.order_by(sim, asc=False)
        .select(custom_memory.content, score=sim)
        .limit(2)
        .collect()
    )
    
    print("Custom dimensions search results:")
    for result in results:
        print(f"Score: {result['score']:.4f} - {result['content']}")
        
except Exception as e:
    print(f"Custom dimensions example failed: {e}")

# Example 4: Direct embedding access (if needed for analysis)
print("\n=== Example 4: Direct Embedding Access ===")

try:
    # Access raw embeddings for analysis
    embeddings_data = (
        memory_openai.select(
            memory_openai.doc_id,
            memory_openai.content,
            embedding_vector=memory_openai.content.embedding()
        )
        .limit(2)
        .collect()
    )
    
    print("Direct embedding access:")
    for row in embeddings_data:
        print(f"Doc: {row['doc_id']}")
        print(f"Embedding shape: {row['embedding_vector'].shape}")
        print(f"First 5 values: {row['embedding_vector'][:5]}")
        print()
        
except Exception as e:
    print(f"Direct embedding access failed: {e}")

# Example 5: Practical use case - Document similarity clustering
print("=== Example 5: Document Similarity Analysis ===")

# Find most similar document pairs
print("Finding most similar document pairs:")

all_docs_result = memory_openai.select(
    memory_openai.doc_id,
    memory_openai.content, 
    memory_openai.category
).collect()

# Convert to Python list for easier manipulation
all_docs = list(all_docs_result)

# Compare each document with others
for i, doc1 in enumerate(all_docs):
    for j in range(i + 1, len(all_docs)):
        doc2 = all_docs[j]
        
        # Use doc1 content to search for similarity with doc2
        sim_score = (
            memory_openai.where(memory_openai.doc_id == doc2['doc_id'])
            .select(
                similarity=memory_openai.content.similarity(doc1['content'])
            )
            .collect()
        )
        
        if sim_score:
            score = sim_score[0]['similarity']
            print(f"{doc1['doc_id']} ↔ {doc2['doc_id']}: {score:.4f}")
            print(f"  Categories: {doc1['category']} ↔ {doc2['category']}")