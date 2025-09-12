#!/usr/bin/env python3
"""
Interactive CLI for the Infinite Memory LangChain Agent

Experience persistent AI memory in action!
"""

import os
import sys
import argparse
from datetime import datetime
from agent import InfiniteMemoryAgent

class AgentCLI:
    """Interactive command-line interface for the memory agent"""
    
    def __init__(self):
        """Initialize the CLI"""
        self.agent = None
        self.current_user = None
        self.current_session = None
        
        # Check for API key
        if not os.getenv("OPENAI_API_KEY"):
            print("❌ OPENAI_API_KEY not found!")
            print("Please set your API key: export OPENAI_API_KEY='your-key'")
            sys.exit(1)
    
    def start(self):
        """Start the interactive CLI"""
        
        print("\n" + "=" * 70)
        print("🧠 INFINITE MEMORY LANGCHAIN AGENT")
        print("=" * 70)
        print("An AI that remembers everything you tell it!")
        print("\nCommands:")
        print("  /help     - Show all commands")
        print("  /stats    - Show memory statistics") 
        print("  /search   - Search through all memories")
        print("  /profile  - View user profile")
        print("  /history  - View conversation history")
        print("  /user     - Switch user")
        print("  /exit     - Exit the program")
        print("\nJust type normally to chat!")
        print("=" * 70)
        
        # Initialize agent
        try:
            self.agent = InfiniteMemoryAgent("MemoryBot")
            print("✅ Agent initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize agent: {e}")
            return
        
        # Get user identity
        self._setup_user()
        
        # Start chat loop
        self._chat_loop()
    
    def _setup_user(self):
        """Set up user identity"""
        
        print(f"\n👤 User Setup")
        print("-" * 20)
        
        while not self.current_user:
            user_input = input("What's your name? (or user ID): ").strip()
            if user_input:
                self.current_user = user_input
                self.current_session = f"{user_input}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                print(f"✅ Welcome {self.current_user}!")
                print(f"📝 Session: {self.current_session}")
                break
    
    def _chat_loop(self):
        """Main chat interaction loop"""
        
        print(f"\n💬 Start chatting with {self.agent.agent_name}!")
        print("Type /help for commands or just chat normally...\n")
        
        while True:
            try:
                user_input = input(f"{self.current_user}: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith('/'):
                    if not self._handle_command(user_input):
                        break
                    continue
                
                # Regular chat
                print(f"{self.agent.agent_name}: ", end="", flush=True)
                
                try:
                    response = self.agent.chat(
                        user_id=self.current_user,
                        message=user_input,
                        session_id=self.current_session
                    )
                    print(response)
                    
                except Exception as e:
                    print(f"Sorry, I encountered an error: {e}")
                
                print()  # Add space between exchanges
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except EOFError:
                print("\n\n👋 Goodbye!")
                break
    
    def _handle_command(self, command: str) -> bool:
        """Handle CLI commands. Returns False to exit."""
        
        cmd = command.lower().strip()
        
        if cmd == '/help':
            self._show_help()
        
        elif cmd == '/stats':
            self._show_stats()
        
        elif cmd.startswith('/search'):
            query = command[7:].strip() if len(command) > 7 else None
            self._search_memories(query)
        
        elif cmd == '/profile':
            self._show_profile()
        
        elif cmd == '/history':
            self._show_history()
        
        elif cmd == '/user':
            self._switch_user()
        
        elif cmd == '/exit' or cmd == '/quit':
            print("👋 Goodbye!")
            return False
        
        else:
            print(f"❓ Unknown command: {command}")
            print("Type /help for available commands")
        
        return True
    
    def _show_help(self):
        """Show help information"""
        
        print("\n📚 Available Commands:")
        print("-" * 30)
        print("/help          - Show this help")
        print("/stats         - Memory statistics")
        print("/search <text> - Search memories")
        print("/profile       - View user profile") 
        print("/history       - Recent conversation history")
        print("/user          - Switch to different user")
        print("/exit          - Exit the program")
        print("\n💡 Just type normally to chat with the agent!")
    
    def _show_stats(self):
        """Show memory statistics"""
        
        print(f"\n📊 Memory Statistics for {self.current_user}")
        print("-" * 40)
        
        try:
            stats = self.agent.get_memory_stats(self.current_user)
            
            print(f"Total memories: {stats['total_memories']}")
            print(f"Topics discussed: {', '.join(stats['topics'].keys())}")
            print("Topic distribution:")
            for topic, count in stats['topics'].items():
                print(f"  {topic}: {count} messages")
            
            print("\nImportance distribution:")
            for level, count in stats['importance_distribution'].items():
                print(f"  {level}: {count} messages")
                
        except Exception as e:
            print(f"❌ Could not retrieve stats: {e}")
    
    def _search_memories(self, query: str = None):
        """Search through memories"""
        
        if not query:
            query = input("🔍 Search for: ").strip()
        
        if not query:
            print("❓ No search query provided")
            return
        
        print(f"\n🔍 Searching memories for: '{query}'")
        print("-" * 50)
        
        try:
            results = self.agent.search_all_memories(query, self.current_user, limit=5)
            
            if results:
                for i, result in enumerate(results, 1):
                    date = result['timestamp'][:10]
                    print(f"  {i}. [{date}] {result['role']}: {result['content'][:80]}...")
                    print(f"     Topic: {result['topic']}, Similarity: {result['similarity']:.3f}")
                    print()
            else:
                print("❌ No memories found matching your search")
                
        except Exception as e:
            print(f"❌ Search failed: {e}")
    
    def _show_profile(self):
        """Show user profile"""
        
        print(f"\n👤 Profile for {self.current_user}")
        print("-" * 30)
        
        profile_info = self.agent._get_user_profile(self.current_user)
        print(profile_info)
    
    def _show_history(self):
        """Show recent conversation history"""
        
        print(f"\n📜 Recent History for {self.current_user}")
        print("-" * 40)
        
        try:
            recent = (
                self.agent.memory
                .where(self.agent.memory.user_id == self.current_user)
                .order_by(self.agent.memory.timestamp, asc=False)
                .select(
                    self.agent.memory.role,
                    self.agent.memory.content,
                    self.agent.memory.timestamp,
                    self.agent.memory.topic
                )
                .limit(10)
                .collect()
            )
            
            for memory in recent:
                date_time = memory['timestamp'][:16].replace('T', ' ')
                print(f"[{date_time}] {memory['role']}: {memory['content'][:60]}...")
                print(f"   Topic: {memory['topic']}")
                print()
                
        except Exception as e:
            print(f"❌ Could not retrieve history: {e}")
    
    def _switch_user(self):
        """Switch to a different user"""
        
        print(f"\n🔄 Switch User")
        print("-" * 20)
        
        new_user = input("Enter new user name/ID: ").strip()
        if new_user:
            self.current_user = new_user
            self.current_session = f"{new_user}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            print(f"✅ Switched to user: {self.current_user}")
            print(f"📝 New session: {self.current_session}")
        else:
            print("❌ No user provided")


def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(description="Infinite Memory LangChain Agent CLI")
    parser.add_argument("--user", help="User ID to start with", default=None)
    args = parser.parse_args()
    
    cli = AgentCLI()
    
    if args.user:
        cli.current_user = args.user
        cli.current_session = f"{args.user}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    cli.start()


if __name__ == "__main__":
    main()
