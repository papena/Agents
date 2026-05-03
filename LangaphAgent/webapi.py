from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Dict, Any
from VectorDB import vectorstore

app = FastAPI()


# ---- Request Schema ----
class EmployeeSearchRequest(BaseModel):
    query: str
    top_k: int = 5
    location: Optional[str] = None
    role: Optional[str] = None


# ---- Your Core Tool ----
def search_employees_tool(query: str, top_k: int = 5, filters: Dict[str, Any] = None):

    print("Request received in  search_employees_tool")
    print(query)
    print(top_k)
    print(filters)
    results = vectorstore.similarity_search_with_score(
        query=query,
        k=top_k,
        filter=filters
    )

    output = []

    for doc, distance in results:
        metadata = doc.metadata or {}

        similarity = 1 - distance  # assuming cosine

        output.append({
            "id": metadata.get("id"),
            "name": metadata.get("name"),
            "role": metadata.get("role"),
            "location": metadata.get("location"),
            "score": round(similarity, 4),
            "content": doc.page_content
        })

    return {"results": output}


# ---- MCP Endpoint ----
@app.post("/tools/employee_search")
def employee_search(req: EmployeeSearchRequest):

    conditions = []
    
    if req.location:
        conditions.append({"location": {"$eq": req.location}})

    if req.role:
        conditions.append({"role": {"$eq": req.role}})

    # ✅ Build valid Chroma filter
    if len(conditions) == 1:
        filters = conditions[0]
    elif len(conditions) > 1:
        filters = {"$and": conditions}
    else:
        filters = None

    return search_employees_tool(
        query=req.query,
        top_k=req.top_k,
        filters=filters
    )


@app.get("/tools")
def list_tools():
    return {
        "tools": [
            {
                "name": "employee_search",
                "description": "Search employees by skills, role, and location.Use this tool to search employees by skills, role, or location.Always use it when user asks about employees.",
                "endpoint": "/tools/employee_search",
                "method": "POST",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "location": {"type": "string"},
                        "role": {"type": "string"},
                        "top_k": {"type": "integer"}
                    },
                    "required": ["query"]
                },
                "output_schema": {
                    "type": "object",
                    "properties": {
                        "results": {"type": "array"}
                    }
                }
            }
        ]
    }

import uvicorn

if __name__ == "__main__":
    uvicorn.run("webapi:app", host="0.0.0.0", port=8000, reload=True)