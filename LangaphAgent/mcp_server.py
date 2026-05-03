from mcp.server.fastmcp import FastMCP
import requests

mcp = FastMCP("Employee Tools")


@mcp.tool()
def employee_search(query: str, location: str = None, role: str = None):
    """Search employees via your FastAPI backend"""
    print("Tool called with:", query, location, role)

    try:
        response = requests.post(
            "http://localhost:8000/tools/employee_search",
            json={
                "query": query,
                "location": location,
                "role": role,
                "top_k": 5
            },
            timeout=5
        )

        response.raise_for_status()  # ✅ catch HTTP errors

        return {
            "success": True,
            "data": response.json()
        }

    except Exception as e:
        print("ERROR in tool:", str(e))

        return {
            "success": False,
            "error": str(e)
        }

if __name__ == "__main__":
    print("Starting MCP server...")
    # mcp.run()
    mcp.run(transport="stdio")