from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="The Mask Core System",
    description="মূল AI অ্যাসিস্টেন্ট - বাংলায় কথা বলে",
    version="1.0.0"
)

# লোকাল Ollama মডেল
llm = ChatOllama(model="qwen2.5-coder:14b", base_url="http://localhost:11434")

class ChatRequest(BaseModel):
    message: str

@app.get("/")
def root():
    return {"message": "The Mask Core System চালু আছে! বাংলায় কথা বলতে পারি।"}

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        response = llm.invoke([HumanMessage(content=request.message)])
        return {"response": response.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)