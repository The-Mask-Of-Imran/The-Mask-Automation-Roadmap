from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
import importlib.util
import sys

app = FastAPI(
    title="The Mask Automation - Core System",
    description="কোর সিস্টেম যা সব প্লাগিন লোড করে এবং কমান্ড রান করে",
    version="1.0.0"
)

# অফলাইন LLM (Ollama) – Render-এ পরে external Ollama ব্যবহার করব
llm = ChatOllama(model="qwen2.5-coder:7b")

# প্লাগিন লোডার ফাংশন
def load_plugin(plugin_name):
    try:
        # প্লাগিন ফাইল লোড (যদি লোকাল থাকে – পরে GitHub থেকে ডাউনলোড লজিক যোগ করব)
        spec = importlib.util.spec_from_file_location(plugin_name, f"plugins/{plugin_name}.py")
        if spec is None:
            return None
        module = importlib.util.module_from_spec(spec)
        sys.modules[plugin_name] = module
        spec.loader.exec_module(module)
        print(f"Loaded plugin: {plugin_name}")
        return module
    except Exception as e:
        print(f"Error loading plugin {plugin_name}: {e}")
        return None

# উদাহরণ: কয়েকটা প্লাগিন লোড (পরে অটো লিস্ট থেকে লোড করব)
loaded_plugins = {}
plugins_list = [
    "The-Mask-Automation-Feature-Analysis-Plugin",
    "The-Mask-Automation-Roadmap-Generator-Plugin",
    # তোমার ১৫টা প্লাগিনের নাম যোগ করো
]

for p in plugins_list:
    loaded_plugins[p] = load_plugin(p)

class CommandRequest(BaseModel):
    command: str

@app.get("/")
def root():
    return {
        "message": "The Mask Core System is running!",
        "status": "active",
        "loaded_plugins": list(loaded_plugins.keys()),
        "version": "1.0.0"
    }

@app.post("/command")
async def run_command(request: CommandRequest):
    try:
        # এখানে পরে প্লাগিন লোড করে কমান্ড রান করবে
        response = llm.invoke([HumanMessage(content=request.command)])
        return {"response": response.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upgrade")
async def self_upgrade():
    # পরে এখানে git pull + restart লজিক যোগ করব
    return {"status": "Upgrade initiated (placeholder)"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)