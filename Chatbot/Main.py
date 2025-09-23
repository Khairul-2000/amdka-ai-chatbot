from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
# from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.memory import MemorySaver
from Mongodb import MongoDBSaver

# from langgraph.checkpoint.postgres import PostgresSaver
import uuid
from config import AIState, tools
from Nodes import agent_node, tool_output_node



# saver = PostgresSaver.from_conn_string("postgresql://amdka_database_ui6s_user:I5yzse0ifqKIAce0AG4knOhoC8tOoIbt@dpg-d3981l1r0fns738g3pqg-a.oregon-postgres.render.com/amdka_database_ui6s")

# ---------------------------
# Build Graph
# ---------------------------
graph = StateGraph(AIState)

graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(tools))


graph.add_node("tool_output", tool_output_node)

graph.add_conditional_edges(
    "agent",
    lambda state: "tools" if state["messages"][-1].tool_calls else END,
    {"tools": "tools", END: END}
)

graph.add_edge("tools", "tool_output")
graph.add_edge("tool_output", "agent")

graph.set_entry_point("agent")




# ---------------------------
# Memory Setup
# ---------------------------
# Create memory saver for persistent conversations
# memory = MemorySaver()

mongodb = MongoDBSaver()

# Compile app with memory
app = graph.compile(checkpointer=mongodb)



# ---------------------------
# Example usage with Memory
# ---------------------------
if __name__ == "__main__":
    # Create a unique thread ID for this conversation
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    print(f"🧠 Testing with Memory (Session: {thread_id[:8]})")
    print("="*60)

    Option = True
    while Option:
        
        user_input = input("👤 You: ")
        if user_input.lower() in ['quit', 'exit', 'bye']:
            break
            
        # First message
        messages = [HumanMessage(content=user_input)]
      
        print("\n" + "="*50 + "\n")

        result = app.invoke({"messages": [("user", messages[0])]}, config=config)

        print("🤖 Bot Response (JSON):")
        # Get the final AI response (last message that's not a tool call response)
        for msg in reversed(result["messages"]):
            if (hasattr(msg, 'content') and 
                type(msg).__name__ == 'AIMessage' and 
                not msg.content.startswith('{"success"') and
                not msg.content.startswith('Found')):
                try:
                    # Try to parse and pretty print JSON
                    import json
                    response_json = json.loads(msg.content)
                    print(json.dumps(response_json, indent=2))
                except:
                    # If it's not valid JSON, print as is
                    print(msg.content)
                break
        
     