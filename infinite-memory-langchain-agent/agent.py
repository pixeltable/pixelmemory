#!/usr/bin/env python3
"""
Infinite Memory LangChain Agent

A stateful AI agent with persistent memory powered by Pixelmemory.
Experience what it's like to have an AI that truly remembers everything.
"""

from pixelmemory import Memory
from pixelmemory.context import Text
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from datetime import datetime
from typing import List, Dict, Optional
import uuid
import os

class InfiniteMemoryAgent:
    """An AI agent that never forgets anything"""
    
    def __init__(self, agent_name: str = "InfiniteAgent"):
        """Initialize the agent with infinite memory"""
        
        self.agent_name = agent_name
        self.llm = init_chat_model("gpt-4o-mini", model_provider="openai")
        
        # Set up comprehensive memory system
        memory_context = [
            Text(id="user_id", embed=False),          # Who is talking
            Text(id="session_id", embed=False),       # Current session
            Text(id="role", embed=False),             # user/assistant  
            Text(id="content", embed=True),           # Searchable message content
            Text(id="topic", embed=False),            # Conversation topic
            Text(id="timestamp", embed=False),        # When it happened
            Text(id="importance", embed=False),       # High/medium/low importance
        ]
        
        self.memory = Memory(
            context=memory_context,
            namespace="infinite_agent",
            table_name="all_conversations",
            if_exists="ignore"
        )
        
        # User profile memory
        profile_context = [
            Text(id="user_id", embed=False),
            Text(id="name", embed=False),
            Text(id="interests", embed=True),         # Searchable interests
            Text(id="preferences", embed=False),
            Text(id="background", embed=True),        # Searchable background info
            Text(id="last_seen", embed=False),
        ]
        
        self.profile_memory = Memory(
            context=profile_context,
            namespace="infinite_agent", 
            table_name="user_profiles",
            if_exists="ignore"
        )
        
        print(f"🧠 {agent_name} initialized with infinite memory")
        print("💭 I will remember everything we discuss...")

    def chat(self, user_id: str, message: str, session_id: str = None) -> str:
        """Chat with infinite memory"""
        
        session_id = session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        timestamp = datetime.now().isoformat()
        
        # 1. Store user message
        user_entry = self.memory.Entry(
            user_id=user_id,
            session_id=session_id,
            role="user",
            content=message,
            topic=self._extract_topic(message),
            timestamp=timestamp,
            importance=self._assess_importance(message)
        )
        self.memory.add(user_entry)
        
        # 2. Get user profile
        user_profile = self._get_user_profile(user_id)
        
        # 3. Search for relevant memories
        relevant_memories = self._search_relevant_memories(user_id, message)
        
        # 4. Build comprehensive context
        memory_context = self._build_memory_context(relevant_memories)
        
        # 5. Create system prompt with full context
        system_prompt = f"""You are {self.agent_name}, an AI with infinite memory. You remember every conversation, every detail, and every interaction.

USER PROFILE:
{user_profile}

RELEVANT MEMORIES FROM PAST CONVERSATIONS:
{memory_context}

CURRENT SESSION: {session_id}
TIMESTAMP: {timestamp}

Instructions:
- Use your infinite memory to provide contextual, personalized responses
- Reference past conversations when relevant
- Show that you remember details about the user
- Be conversational and build on your relationship with the user
- If this is a new user, be welcoming and start learning about them
"""
        
        # 6. Generate response
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=message)
        ]
        
        response = self.llm.invoke(messages)
        ai_response = response.content
        
        # 7. Store AI response
        ai_entry = self.memory.Entry(
            user_id=user_id,
            session_id=session_id,
            role="assistant",
            content=ai_response,
            topic=self._extract_topic(ai_response),
            timestamp=datetime.now().isoformat(),
            importance="medium"
        )
        self.memory.add(ai_entry)
        
        # 8. Update user profile if we learned something new
        self._update_user_profile_from_message(user_id, message)
        
        return ai_response

    def _extract_topic(self, message: str) -> str:
        """Extract topic from message (simple keyword-based)"""
        topics = {
            "programming": ["python", "code", "coding", "programming", "software", "development"],
            "ai_ml": ["ai", "machine learning", "ml", "data science", "neural", "model"],
            "work": ["job", "work", "career", "office", "project", "business"],
            "personal": ["family", "friend", "hobby", "weekend", "vacation", "personal"],
            "tech": ["technology", "computer", "software", "hardware", "tech"],
            "learning": ["learn", "study", "course", "education", "tutorial", "book"]
        }
        
        message_lower = message.lower()
        for topic, keywords in topics.items():
            if any(keyword in message_lower for keyword in keywords):
                return topic
        return "general"

    def _assess_importance(self, message: str) -> str:
        """Assess message importance"""
        important_indicators = ["important", "urgent", "help", "problem", "issue", "stuck"]
        personal_indicators = ["my name is", "i am", "i'm", "personal", "family"]
        
        message_lower = message.lower()
        
        if any(indicator in message_lower for indicator in important_indicators):
            return "high"
        elif any(indicator in message_lower for indicator in personal_indicators):
            return "high"
        else:
            return "medium"

    def _search_relevant_memories(self, user_id: str, message: str, limit: int = 5) -> List[Dict]:
        """Search for relevant memories"""
        
        try:
            similarity = self.memory.content.similarity(message)
            results = (
                self.memory
                .where((self.memory.user_id == user_id) & (similarity >= 0.3))
                .order_by(similarity, asc=False)
                .select(
                    self.memory.role,
                    self.memory.content,
                    self.memory.topic,
                    self.memory.timestamp,
                    self.memory.importance,
                    similarity=similarity
                )
                .limit(limit)
                .collect()
            )
            return [dict(r) for r in results]
        except:
            return []

    def _build_memory_context(self, memories: List[Dict]) -> str:
        """Build formatted memory context"""
        
        if not memories:
            return "No relevant past memories found."
        
        context_lines = []
        for memory in memories:
            timestamp = memory['timestamp'][:10]  # Just date
            context_lines.append(
                f"[{timestamp}] {memory['role']}: {memory['content'][:100]}... "
                f"(topic: {memory['topic']}, importance: {memory['importance']})"
            )
        
        return "\n".join(context_lines)

    def _get_user_profile(self, user_id: str) -> str:
        """Get user profile"""
        
        try:
            profile = (
                self.profile_memory
                .where(self.profile_memory.user_id == user_id)
                .select(
                    self.profile_memory.name,
                    self.profile_memory.interests,
                    self.profile_memory.background,
                    self.profile_memory.preferences
                )
                .collect()
            )
            
            if profile:
                p = profile[0]
                return f"Name: {p['name']}\nInterests: {p['interests']}\nBackground: {p['background']}\nPreferences: {p['preferences']}"
            else:
                return "New user - no profile yet."
        except:
            return "No profile available."

    def _update_user_profile_from_message(self, user_id: str, message: str):
        """Auto-update user profile from conversation"""
        
        msg_lower = message.lower()
        
        # Extract name
        if "my name is" in msg_lower or "i'm " in msg_lower or "i am " in msg_lower:
            # Simple name extraction (in production, use NLP)
            words = message.split()
            if "name is" in message.lower():
                idx = next(i for i, word in enumerate(words) if word.lower() == "is")
                if idx + 1 < len(words):
                    name = words[idx + 1].strip(".,!?")
                    self._upsert_profile(user_id, name=name)
        
        # Extract interests/background  
        if any(word in msg_lower for word in ["work", "job", "engineer", "scientist", "developer"]):
            interests = "technology, programming"
            background = "technical professional"
            self._upsert_profile(user_id, interests=interests, background=background)

    def _upsert_profile(self, user_id: str, **kwargs):
        """Update or insert user profile"""
        
        try:
            # Get existing profile
            existing = self.profile_memory.where(self.profile_memory.user_id == user_id).collect()
            
            if existing:
                profile = existing[0]
                # Update with new info
                updated_entry = self.profile_memory.Entry(
                    user_id=user_id,
                    name=kwargs.get('name', profile.get('name', '')),
                    interests=kwargs.get('interests', profile.get('interests', '')),
                    background=kwargs.get('background', profile.get('background', '')),
                    preferences=kwargs.get('preferences', profile.get('preferences', '')),
                    last_seen=datetime.now().isoformat()
                )
                # Delete old and add new
                self.profile_memory.delete(self.profile_memory.user_id == user_id)
                self.profile_memory.add(updated_entry)
            else:
                # Create new profile
                new_entry = self.profile_memory.Entry(
                    user_id=user_id,
                    name=kwargs.get('name', ''),
                    interests=kwargs.get('interests', ''),
                    background=kwargs.get('background', ''),
                    preferences=kwargs.get('preferences', ''),
                    last_seen=datetime.now().isoformat()
                )
                self.profile_memory.add(new_entry)
        except Exception as e:
            print(f"Profile update failed: {e}")

    def get_memory_stats(self, user_id: str = None) -> Dict:
        """Get comprehensive memory statistics"""
        
        try:
            if user_id:
                memories = self.memory.where(self.memory.user_id == user_id).collect()
            else:
                memories = self.memory.collect()
            
            topics = {}
            importance_counts = {"high": 0, "medium": 0, "low": 0}
            
            for memory in memories:
                # Count topics
                topic = memory.get('topic', 'general')
                topics[topic] = topics.get(topic, 0) + 1
                
                # Count importance
                importance = memory.get('importance', 'medium')
                importance_counts[importance] = importance_counts.get(importance, 0) + 1
            
            return {
                "total_memories": len(memories),
                "topics": topics,
                "importance_distribution": importance_counts,
                "user_specific": bool(user_id)
            }
        except:
            return {"error": "Could not retrieve memory stats"}

    def search_all_memories(self, query: str, user_id: str = None, limit: int = 10) -> List[Dict]:
        """Search all memories"""
        
        try:
            similarity = self.memory.content.similarity(query)
            search_query = self.memory.where(similarity >= 0.4)
            
            if user_id:
                search_query = search_query.where(self.memory.user_id == user_id)
            
            results = (
                search_query
                .order_by(similarity, asc=False)
                .select(
                    self.memory.user_id,
                    self.memory.role,
                    self.memory.content,
                    self.memory.topic,
                    self.memory.timestamp,
                    similarity=similarity
                )
                .limit(limit)
                .collect()
            )
            
            return [dict(r) for r in results]
        except:
            return []
