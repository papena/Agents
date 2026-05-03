import asyncio
import os
import sys

from langchain_groq import ChatGroq
from langchain_core.tools import StructuredTool
from langchain_core.messages import HumanMessage, SystemMessage

from langgraph.graph import START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp import ClientSession


# -------------------------------
# ✅ LLM
# -------------------------------
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key="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"  # ⚠️ don't hardcode in real apps
)


# -------------------------------
# ✅ Create MCP Tool (ASYNC SAFE)
# -------------------------------
def create_tool(session, tool_name):

    from pydantic import BaseModel, Field

    class ToolInput(BaseModel):
        query: str = Field(..., description="Search query")
        location: str | None = None
        role: str | None = None
        skill: str | None = None

    async def _call_async(*args, **kwargs):
        # 🔥 Handle LangChain input formats

        if args and isinstance(args[0], dict):
            payload = args[0]
        else:
            payload = kwargs

        print("📤 Sending to MCP:", payload)  # DEBUG

        return await session.call_tool(tool_name, payload)

    return StructuredTool.from_function(
        coroutine=_call_async,
        name=tool_name,
        description=f"MCP tool: {tool_name}",
        args_schema=ToolInput
    )


# -------------------------------
# ✅ MAIN
# -------------------------------
async def main():

    params = StdioServerParameters(
        command="python",
        args=[os.path.join(os.path.dirname(__file__), "mcp_server.py")]
    )

    print("🚀 Starting MCP client...")

    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:

            print("✅ Initializing MCP session...")
            await session.initialize()

            print("📦 Fetching MCP tools...")
            result = await session.list_tools()
            mcp_tool_defs = result.tools

            print("🛠 Available tools:", [t.name for t in mcp_tool_defs])

            # ✅ Create live tools bound to active session
            tools = [create_tool(session, t.name) for t in mcp_tool_defs]

            # -------------------------------
            # ✅ Agent Setup
            # -------------------------------
            sys_msg = SystemMessage(
                content="""You are a helpful assistant responsible to get employee details.
                
                When calling tools:
                - If tool returns success=false, DO NOT retry the same call
                - Instead, explain the issue or ask user for clarification."""
            )

            llm_with_tools = llm.bind_tools(tools)

            async def assistant(state: MessagesState):
                response = await llm_with_tools.ainvoke(
                    [sys_msg] + state["messages"]
                )
                return {"messages": [response]}

            # -------------------------------
            # ✅ LangGraph
            # -------------------------------
            builder = StateGraph(MessagesState)

            builder.add_node("assistant", assistant)
            builder.add_node("tools", ToolNode(tools))

            builder.add_edge(START, "assistant")
            builder.add_conditional_edges("assistant", tools_condition)
            builder.add_edge("tools", "assistant")

            graph = builder.compile()

            # -------------------------------
            # ✅ Run Agent
            # -------------------------------
            print("🤖 Running agent...\n")

            result = await graph.ainvoke({
                "messages": [
                    HumanMessage(content="Find NLP Data Scientist in Bangalore")
                ]
            })

            for msg in result["messages"]:
                msg.pretty_print()


# -------------------------------
# ✅ ENTRY
# -------------------------------
if __name__ == "__main__":
    asyncio.run(main())