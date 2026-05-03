
import datetime
from langchain_groq import ChatGroq

# llm = ChatOpenAI(model="gpt-4o")
llm = ChatGroq(
    model="llama-3.3-70b-versatile"
)

def shopopeninghours() -> str:
    "This tool provides the daily opening hours schedule for a shop."
    schedule = """Monday 10:00am to 08:00pm
                  Tuesday 10:00am to 08:00pm
                  Wednesday 10:00am to 08:00pm
                  Thursday 10:00pm to 11:00pm
                  Friday 10:00am to 08:00pm
                  Saturday 03:00pm to 05:00pm
                  Sunday 1:00pm to 06:00pm"""
    return schedule

def Currenttime() -> datetime:
    "This tool provides the current date and time with day name."
    return datetime.datetime.now().strftime("%A, %B %d, %Y %H:%M:%S")
    # //return datetime.datetime.now()

tools = [shopopeninghours,Currenttime]

llm_with_tools = llm.bind_tools(tools)

from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, SystemMessage

sys_msg = SystemMessage(content="""You are a helpful assistant responsible 
                                  for determining whether a shop is open or closed 
                                  based on the current time and the shop's opening schedule.""")

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

messages = [HumanMessage(content="Tell me if the shop is open today ")]
messages = react_graph.invoke({"messages" : messages})

for m in messages["messages"]:
    m.pretty_print()

    


