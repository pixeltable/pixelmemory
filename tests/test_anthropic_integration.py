"""
Tests for the Anthropic API integration.

This file contains tests for the Anthropic API integration in the chat_with_memories.py example.
"""

import unittest
import os
from anthropic import Anthropic

class TestAnthropicIntegration(unittest.TestCase):
    """Test cases for the Anthropic API integration."""

    def setUp(self):
        """Set up test fixtures."""
        # Initialize Anthropic client
        self.anthropic_client = Anthropic()
    
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
    
    def test_anthropic_memory_integration(self):
        """Test the integration of Anthropic with memory context."""
        # Skip this test if no API key is set
        if not os.environ.get("ANTHROPIC_API_KEY"):
            self.skipTest("ANTHROPIC_API_KEY environment variable not set")
        
        # Simulate memory context
        memories = [
            "user: My favorite color is blue. assistant: I'll remember that your favorite color is blue.",
            "user: I live in Paris. assistant: I'll remember that you live in Paris."
        ]
        memories_str = "\n".join(f"- {memory}" for memory in memories)
        
        # Generate Assistant response with memory context
        system_prompt = f"You are a helpful AI. Answer the question based on query and memories.\nUser Memories:\n{memories_str}"
        
        response = self.anthropic_client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=100,
            system=system_prompt,
            messages=[{"role": "user", "content": "What is my favorite color?"}]
        )
        
        assistant_response = response.content[0].text
        
        # Check that the response mentions blue
        self.assertIn("blue", assistant_response.lower())
        
        # Test another query
        response = self.anthropic_client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=100,
            system=system_prompt,
            messages=[{"role": "user", "content": "Where do I live?"}]
        )
        
        assistant_response = response.content[0].text
        
        # Check that the response mentions Paris
        self.assertIn("paris", assistant_response.lower())


# Simple test runner
if __name__ == "__main__":
    unittest.main()
