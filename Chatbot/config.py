from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages
from Tools import product_search
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if api_key is None:
    raise ValueError("OPENAI_API_KEY environment variable not set")

class AIState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

tools = [product_search]
model = ChatOpenAI(model="gpt-4o", api_key=api_key).bind_tools(tools)