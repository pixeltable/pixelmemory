"""
Tests for the Memory class.

This file contains all tests for the Memory class in pixelmemory/memory.py.
Uses Anthropic's Claude API for real API calls.
"""

import unittest
import datetime
import json
import os
from anthropic import Anthropic
from pixelmemory.memory import Memory


class TestMemory(unittest.TestCase):
    """Test cases for the Memory class."""

    def setUp(self):
        """Set up test fixtures."""
        # Initialize the Memory object with the default embedding model
        self.memory = Memory()
        
        # Initialize Anthropic client
        self.anthropic_client = Anthropic()
    
    def test_init(self):
        """Test the initialization of the Memory object."""
        # Check that the embedding model was initialized correctly
        self.assertIsNotNone(self.memory.embedding_model)
    
    def test_add_and_search(self):
        """Test adding messages to memory and searching for them."""
        # Generate a unique user ID for this test (using only alphanumeric characters)
        user_id = f"test_user_{int(datetime.datetime.now().timestamp())}"
        
        # Test data
        messages = [
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris."}
        ]
        
        # Add the messages to memory
        self.memory.add(messages, user_id=user_id)
        
        # Search for the messages
        results = self.memory.search("capital of France", user_id=user_id, limit=3)
        
        # Check that we got results
        self.assertTrue(len(results["results"]) > 0)
        
        # Check that the results contain our message
        found = False
        for result in results["results"]:
            if "france" in result["memory"].lower():
                found = True
                break
        self.assertTrue(found)
    
    def test_anthropic_api(self):
        """Test that the Anthropic API is working."""
        # Skip this test if no API key is set
        if not os.environ.get("ANTHROPIC_API_KEY"):
            self.skipTest("ANTHROPIC_API_KEY environment variable not set")
        
        # Test the Anthropic API
        response = self.anthropic_client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=100,
            system="You are a helpful assistant.",
            messages=[{"role": "user", "content": "What is 2+2? Answer with just the number."}]
        )
        
        # Check that we got a response
        self.assertIsNotNone(response)
        self.assertIsNotNone(response.content)
        self.assertTrue(len(response.content) > 0)
        
        # Check that the response contains the expected answer
        answer = response.content[0].text.strip()
        self.assertEqual(answer, "4")
    
    def test_chat_with_memory(self):
        """Test the chat with memory functionality."""
        # Skip this test if no API key is set
        if not os.environ.get("ANTHROPIC_API_KEY"):
            self.skipTest("ANTHROPIC_API_KEY environment variable not set")
        
        # Generate a unique user ID for this test (using only alphanumeric characters)
        user_id = f"test_user_{int(datetime.datetime.now().timestamp())}"
        
        # Add some initial memories
        initial_messages = [
            {"role": "user", "content": "My favorite color is blue."},
            {"role": "assistant", "content": "I'll remember that your favorite color is blue."}
        ]
        self.memory.add(initial_messages, user_id=user_id)
        
        # Simulate a chat with memory
        query = "What is my favorite color?"
        relevant_memories = self.memory.search(query=query, user_id=user_id, limit=3)
        memories_str = "\n".join(f"- {entry['memory']}" for entry in relevant_memories["results"])
        
        # Generate Assistant response
        system_prompt = f"You are a helpful AI. Answer the question based on query and memories.\nUser Memories:\n{memories_str}"
        
        response = self.anthropic_client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=100,
            system=system_prompt,
            messages=[{"role": "user", "content": query}]
        )
        
        assistant_response = response.content[0].text
        
        # Check that the response mentions blue
        self.assertIn("blue", assistant_response.lower())


# Simple test runner
if __name__ == "__main__":
    unittest.main()


# Instructions for running tests:
# 
# 1. Using unittest directly:
#    python -m unittest tests/test_memory.py
# 
# 2. Using pytest:
#    pytest tests/test_memory.py
# 
# 3. Running directly:
#    python tests/test_memory.py
