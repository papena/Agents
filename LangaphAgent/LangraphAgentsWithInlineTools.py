
import datetime
from langchain_groq import ChatGroq

# from Directtools import build_tools
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    
import requests

def load_mcp_tools(base_url: str):
    response = requests.get(f"{base_url}/tools")
    tools_data = response.json()["tools"]
    
    return tools_data


# from langchain.tools import StructuredTool
from langchain_core.tools import StructuredTool
from pydantic import create_model

def build_tools(base_url: str):
    tools_data = load_mcp_tools(base_url)
    tools = []

    for t in tools_data:
        name = t["name"]
        description = t["description"]
        schema = t["input_schema"]["properties"]

        # Dynamically create Pydantic model
        fields = {
            k: (str, None) for k in schema.keys()
        }

        InputModel = create_model(f"{name}_input", **fields)

        def make_func(tool_name):
            def _func(**kwargs):
                print(f"Calling tool: {tool_name} with {kwargs}")
                return requests.post(
                    f"{base_url}/tools/{tool_name}",
                    json=kwargs
                ).json()
            return _func

        tool = StructuredTool.from_function(
            func=make_func(name),
            name=name,
            description=description,
            args_schema=InputModel
        )

        tools.append(tool)

    return tools


# llm = ChatOpenAI(model="gpt-4o")
llm = ChatGroq(
    model="llama-3.3-70b-versatile",api_key="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"
)


# tools = [shopopeninghours,Currenttime]
tools = build_tools("http://localhost:8000")

# agent = create_agent(
#     model="gpt-4o-mini",
#     tools=tools
# )
llm_with_tools = llm.bind_tools(tools)

from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, SystemMessage

sys_msg = SystemMessage(content="""You are a helpful assistant responsible 
                                  to get employee details.""")

def assistant(state: MessagesState):
    return {"messages": [llm_with_tools.invoke([sys_msg]+ state["messages"])]}


from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode
from IPython.display import Image, display

builder = StateGraph(MessagesState)

builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")

react_graph = builder.compile()

display(Image(react_graph.get_graph(xray=True).draw_mermaid_png()))

initial_messages = [
    HumanMessage(content="Find NLP developers in Bangalore")
]

# messages = [HumanMessage(content="Tell me if the shop is open today ")]
messages = react_graph.invoke({"messages" : initial_messages})

for m in messages["messages"]:
    m.pretty_print()

