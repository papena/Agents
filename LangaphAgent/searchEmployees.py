from langchain.tools import tool

from LangaphAgent.VectorDB import search_employees_tool

@tool
def search_employees(query: str) -> dict:
    """Search employees using semantic similarity."""
    return search_employees_tool(query)