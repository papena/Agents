from fastapi import FastAPI
from pydantic import BaseModel

from LangraphAgentsMCPToolsUIIntegration import run_agent

app = FastAPI()

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    reply = await run_agent(req.message)
    return {"response": reply}

import uvicorn

if __name__ == "__main__":
    uvicorn.run("agent_api:app", host="0.0.0.0", port=8001, reload=True)