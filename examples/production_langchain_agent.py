#!/usr/bin/env python3
"""
Production LangChain Agent with Pixelmemory Integration

This example demonstrates a stateful AI agent that:
- Remembers conversations across sessions using Pixelmemory
- Uses semantic search to find relevant context
- Integrates seamlessly with LangChain models

Usage:
    pip install pixelmemory langchain langchain-openai
    export OPENAI_API_KEY="your-key-here"
    python production_langchain_agent.py
"""

from pixelmemory import Memory
from pixelmemory.context import Text
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from datetime import datetime
from typing import List, Dict
import uuid

class StatefulAgent:
    """A LangChain agent with persistent memory powered by Pixelmemory"""
    
    def __init__(self, agent_name: str = "assistant"):
        """Initialize the agent with persistent memory"""
        
        self.agent_name = agent_name
        self.llm = init_chat_model("gpt-4o-mini", model_provider="openai")
        
        # Set up conversation memory
        conversation_context = [
            Text(id="session_id", embed=False),      # User session
            Text(id="role", embed=False),            # user/assistant
            Text(id="content", embed=True),          # Searchable content
            Text(id="timestamp", embed=False),       # When message was sent
        ]
        
        self.memory = Memory(
            context=conversation_context,
            namespace=f"agent_{agent_name}",
            table_name="conversations",
            if_exists="ignore"
        )
        
        print(f"✅ Stateful agent '{agent_name}' initialized with persistent memory")

    def chat(self, session_id: str, user_message: str) -> str:
        """Chat with memory integration"""
        
        timestamp = datetime.now().isoformat()
        
        # 1. Store user message
        user_entry = self.memory.Entry(
            session_id=session_id,
            role="user",
            content=user_message,
            timestamp=timestamp
        )
        self.memory.add(user_entry)
        
        # 2. Get relevant conversation history using semantic search
        similarity = self.memory.content.similarity(user_message)
        relevant_history = (
            self.memory
            .where((self.memory.session_id == session_id) & (similarity >= 0.3))
            .order_by(similarity, asc=False)
            .select(self.memory.role, self.memory.content, self.memory.timestamp)
            .limit(5)
            .collect()
        )
        
        # 3. Build context from relevant history
        history_context = ""
        if relevant_history:
            history_lines = []
            for h in relevant_history:
                history_lines.append(f"{h['role']}: {h['content']}")
            history_context = "\n".join(history_lines)
        
        # 4. Create system prompt with memory context
        system_prompt = f"""You are {self.agent_name}, a helpful AI assistant with persistent memory.

RELEVANT CONVERSATION HISTORY:
{history_context if history_context else "No relevant history found."}

Instructions:
- Use the conversation history to maintain context and continuity
- Reference past conversations when helpful
- Be conversational and remember what the user has told you
- Provide helpful, detailed responses
"""
        
        # 5. Generate response with LangChain
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message)
        ]
        
        response = self.llm.invoke(messages)
        ai_response = response.content
        
        # 6. Store AI response
        ai_entry = self.memory.Entry(
            session_id=session_id,
            role="assistant",
            content=ai_response,
            timestamp=datetime.now().isoformat()
        )
        self.memory.add(ai_entry)
        
        return ai_response

    def search_memories(self, query: str, limit: int = 5) -> List[Dict]:
        """Search all memories for specific content"""
        
        similarity = self.memory.content.similarity(query)
        results = (
            self.memory
            .where(similarity >= 0.4)
            .order_by(similarity, asc=False)
            .select(
                self.memory.session_id,
                self.memory.role,
                self.memory.content,
                self.memory.timestamp,
                similarity=similarity
            )
            .limit(limit)
            .collect()
        )
        
        return [dict(result) for result in results]

    def get_session_stats(self, session_id: str) -> Dict:
        """Get conversation statistics for a session"""
        
        messages = self.memory.where(self.memory.session_id == session_id).collect()
        user_msgs = [m for m in messages if m['role'] == 'user']
        ai_msgs = [m for m in messages if m['role'] == 'assistant']
        
        return {
            "total_messages": len(messages),
            "user_messages": len(user_msgs),
            "ai_messages": len(ai_msgs),
            "session_start": messages[0]['timestamp'] if messages else None,
        }


def main():
    """Demonstrate the production agent in action"""
    
    print("=" * 70)
    print("🤖 PRODUCTION LANGCHAIN + PIXELMEMORY AGENT")
    print("=" * 70)
    
    # Create the agent
    agent = StatefulAgent("TechBot")
    
    # Simulate a realistic conversation flow
    session_id = "demo_session_123"
    
    conversations = [
        "Hi! I'm Sarah, a data scientist working on customer churn prediction",
        "I'm struggling with feature selection for my model. What techniques do you recommend?",
        "That's helpful! Can you explain more about recursive feature elimination?", 
        "Perfect! Now I'm curious about handling imbalanced datasets in my churn model",
    ]
    
    print(f"\n🎬 Starting conversation session: {session_id}")
    print("-" * 50)
    
    # Run the conversation
    for i, user_msg in enumerate(conversations, 1):
        print(f"\n💬 Turn {i}")
        print(f"User: {user_msg}")
        
        ai_response = agent.chat(session_id, user_msg)
        print(f"AI: {ai_response}")
        
        if i < len(conversations):
            print("  " + "." * 40)  # Visual separator
    
    # Show session statistics  
    stats = agent.get_session_stats(session_id)
    print(f"\n📊 Session Stats: {stats['total_messages']} total messages")
    
    # Demonstrate memory search
    print(f"\n🔍 Memory Search Demo")
    print("-" * 30)
    
    search_query = "feature selection techniques"
    search_results = agent.search_memories(search_query, limit=3)
    
    print(f"Searching for: '{search_query}'")
    print(f"Found {len(search_results)} relevant memories:")
    
    for i, result in enumerate(search_results, 1):
        print(f"  {i}. [{result['role']}] {result['content'][:60]}...")
        print(f"     Similarity: {result['similarity']:.3f}")
    
    # Demonstrate cross-session memory
    print(f"\n🔄 Cross-Session Memory Demo")
    print("-" * 35)
    
    new_session = "demo_session_456"
    continuation_msg = "Hi again! I'm back to continue working on my churn prediction model"
    
    print(f"\n💬 New Session: {new_session}")
    print(f"User: {continuation_msg}")
    
    ai_response = agent.chat(new_session, continuation_msg)
    print(f"AI: {ai_response}")
    
    print("\n" + "=" * 70)
    print("✅ PRODUCTION DEMO COMPLETE!")
    print("=" * 70)
    
    print("\n🎯 Key Features Demonstrated:")
    print("  ✅ Persistent conversation memory")
    print("  ✅ Semantic search across conversations")
    print("  ✅ Real LangChain LLM integration")  
    print("  ✅ Cross-session memory continuity")
    print("  ✅ Context-aware AI responses")
    print("  ✅ Production-ready memory statistics")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()