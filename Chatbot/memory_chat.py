#!/usr/bin/env python3
"""
Interactive Shopping Bot with Memory
Test the chatbot's memory capabilities across multiple conversation turns
"""

from Main import app
from langchain_core.messages import HumanMessage
import uuid

class MemoryShoppingBot:
    def __init__(self):
        # Generate a unique thread ID for this conversation session
        self.thread_id = str(uuid.uuid4())
        self.config = {"configurable": {"thread_id": self.thread_id}}
        print(f"🧠 Started new conversation with memory (Session ID: {self.thread_id[:8]})")
    
    def chat(self, user_input: str):
        """Send a message to the bot and get response with memory"""
        try:
            # Create user message
            messages = [HumanMessage(content=user_input)]
            
            # Invoke the app with memory config
            result = app.invoke({"messages": messages}, config=self.config)
            
            # Extract and return the final AI response  
            for msg in reversed(result["messages"]):
                if (hasattr(msg, 'content') and 
                    type(msg).__name__ == 'AIMessage' and 
                    not msg.content.startswith('{"success"') and
                    not msg.content.startswith('✨')):
                    return msg.content
            
            return "🤖 I processed your request!"
            
        except Exception as e:
            return f"❌ Error: {e}"
    
    def get_conversation_history(self):
        """Get the current conversation history"""
        try:
            # Get the current state
            state = app.get_state(config=self.config)
            messages = state.values.get("messages", [])
            
            print(f"\n📚 Conversation History ({len(messages)} messages):")
            print("-" * 50)
            
            for i, msg in enumerate(messages, 1):
                if isinstance(msg, HumanMessage):
                    print(f"👤 User: {msg.content}")
                elif hasattr(msg, 'content') and not hasattr(msg, 'tool_calls'):
                    # Skip tool messages, show only AI responses
                    if not msg.content.startswith('{"success"'):
                        print(f"🤖 Bot: {msg.content[:100]}...")
            
            print("-" * 50)
        except Exception as e:
            print(f"❌ Could not retrieve history: {e}")

def main():
    """Interactive chat with memory testing"""
    print("🛒 AI Shopping Bot with Memory")
    print("=" * 50)
    print("💡 Test Memory Features:")
    print("  - Ask about red products, then ask follow-up questions")
    print("  - Mention preferences, then see if bot remembers")
    print("  - Ask about different products across multiple turns")
    print("  - Type 'history' to see conversation memory")
    print("  - Type 'reset' to start a new conversation")
    print("  - Type 'quit' to exit")
    print("=" * 50)
    
    bot = MemoryShoppingBot()
    
    # Sample conversation starters
    print("\n🔥 Suggested test conversation:")
    print("1. 'I'm looking for red products in size M'")
    print("2. 'What about in size L instead?'") 
    print("3. 'I don't like shirts, show me something else'")
    print("4. 'What was the price of that red watch again?'")
    print("5. 'history' (to check memory)")
    print()
    
    conversation_count = 0
    
    while True:
        try:
            user_input = input("💬 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye! Your conversation has been saved in memory.")
                break
            
            if user_input.lower() == 'history':
                bot.get_conversation_history()
                continue
            
            if user_input.lower() == 'reset':
                bot = MemoryShoppingBot()
                conversation_count = 0
                print("🔄 Started fresh conversation with new memory!")
                continue
            
            if not user_input:
                continue
            
            conversation_count += 1
            print(f"\n🤖 Bot (Turn {conversation_count}):")
            response = bot.chat(user_input)
            print(response)
            print()
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()