#!/usr/bin/env python3
"""
Quick demo script to show the infinite memory agent capabilities
"""

from agent import InfiniteMemoryAgent
import time

def run_demo():
    """Run a quick demo of the agent's memory capabilities"""
    
    print("🎬 INFINITE MEMORY AGENT DEMO")
    print("=" * 50)
    
    agent = InfiniteMemoryAgent("DemoBot")
    
    # Simulate a conversation with memory
    conversations = [
        ("alice", "Hi! I'm Alice, a Python developer working on AI applications"),
        ("alice", "I'm particularly interested in machine learning and data science"),
        ("alice", "Can you help me understand neural networks better?"),
        ("bob", "Hello! I'm Bob, a product manager interested in AI for business"),
        ("bob", "How can AI help improve customer experience?"),
        ("alice", "Hi again! Remember me? I wanted to ask more about deep learning"),
    ]
    
    print("\n🎭 Running simulated conversations...")
    print("-" * 40)
    
    for user_id, message in conversations:
        print(f"\n{user_id}: {message}")
        
        response = agent.chat(user_id, message)
        print(f"DemoBot: {response[:100]}...")
        
        # Brief pause for realism
        time.sleep(1)
    
    # Demonstrate memory search
    print(f"\n🔍 MEMORY SEARCH DEMO")
    print("-" * 30)
    
    search_queries = [
        "machine learning",
        "Python development", 
        "business applications"
    ]
    
    for query in search_queries:
        print(f"\nSearching for: '{query}'")
        results = agent.search_all_memories(query, limit=2)
        
        for i, result in enumerate(results, 1):
            print(f"  {i}. [{result['user_id']}] {result['content'][:60]}...")
            print(f"     Similarity: {result['similarity']:.3f}")
    
    # Show statistics
    print(f"\n📊 FINAL STATISTICS")
    print("-" * 25)
    
    alice_stats = agent.get_memory_stats("alice")
    bob_stats = agent.get_memory_stats("bob")
    
    print(f"Alice's memories: {alice_stats['total_memories']}")
    print(f"Bob's memories: {bob_stats['total_memories']}")
    print(f"Alice topics: {list(alice_stats['topics'].keys())}")
    print(f"Bob topics: {list(bob_stats['topics'].keys())}")
    
    print("\n✅ Demo complete! The agent remembers everything!")
    print("💡 Try the interactive CLI: python cli.py")

if __name__ == "__main__":
    run_demo()
