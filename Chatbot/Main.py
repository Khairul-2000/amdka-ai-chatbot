from langchain_core.messages import  HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.postgres import PostgresSaver
from .config import AIState, tools
from .Nodes import agent_node, tool_output_node
import os
from dotenv import load_dotenv
load_dotenv()


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








def main(thread_id: str, user_input: str):
    

    with PostgresSaver.from_conn_string(os.getenv("POSTGRES_URI")) as checkpointer:
        # checkpointer.setup()

    
        # Compile app with the checkpointer
        app = graph.compile(checkpointer=checkpointer)
  
        
        with open("thread_id.txt", "w") as f:
            f.write(thread_id)
        config = {"configurable": {"thread_id":thread_id}}
            
        print(f"🧠 Testing with Memory (Session: {thread_id[:8]})")
        print("="*60)

                # First message
        messages = [HumanMessage(content=user_input)]
            
        print("\n" + "="*50 + "\n")

        result = app.invoke({"messages":messages}, config=config)

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
                    return json.dumps(response_json, indent=2)
                except:
                    # If it's not valid JSON, print as is
                    print(msg.content)
                break

     
           
