from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os
import json
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from datetime import datetime

app = FastAPI(
    title="The Mask Core System",
    description="মূল AI অ্যাসিস্টেন্ট + লং-টার্ম মেমরি",
    version="1.0.1"
)

# লোকাল Ollama মডেল
llm = ChatOllama(model="qwen2.5-coder:14b", base_url="http://localhost:11434")

MEMORY_FILE = "memory.json"

class TaskMemoryManager:
    def __init__(self):
        self.memory = self.load_memory()

    def load_memory(self):
        if os.path.exists(MEMORY_FILE):
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"tasks": {}, "current_task_id": None, "last_updated": str(datetime.now())}

    def save_memory(self):
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(self.memory, f, ensure_ascii=False, indent=2)
        self.memory["last_updated"] = str(datetime.now())

    def start_new_task(self, task_name, initial_data=None):
        task_id = str(len(self.memory["tasks"]) + 1)
        self.memory["tasks"][task_id] = {
            "name": task_name,
            "steps": [],
            "current_step": 0,
            "data": initial_data or {},
            "status": "running"
        }
        self.memory["current_task_id"] = task_id
        self.save_memory()
        return task_id

    def add_step(self, task_id, step_description, result=None):
        if task_id not in self.memory["tasks"]:
            return False
        step = {
            "description": step_description,
            "result": result,
            "timestamp": str(datetime.now())
        }
        self.memory["tasks"][task_id]["steps"].append(step)
        self.memory["tasks"][task_id]["current_step"] += 1
        self.save_memory()
        return True

    def get_current_task(self):
        task_id = self.memory.get("current_task_id")
        if task_id and task_id in self.memory["tasks"]:
            return self.memory["tasks"][task_id]
        return None

class ChatRequest(BaseModel):
    message: str

@app.get("/")
def root():
    return {"message": "The Mask Core System চালু আছে! বাংলায় কথা বলতে পারি।"}

@app.post("/chat")
async def chat(request: ChatRequest):
    memory_manager = TaskMemoryManager()
    current_task = memory_manager.get_current_task()

    prompt = request.message
    if current_task:
        prompt = f"Current task: {current_task['name']}\nPrevious steps: {json.dumps(current_task['steps'], ensure_ascii=False)}\nUser: {request.message}"

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        answer = response.content

        # নতুন টাস্ক শুরু করা যদি প্রয়োজন হয়
        if not current_task and "প্রজেক্ট" in request.message:
            task_id = memory_manager.start_new_task("নতুন প্রজেক্ট")
            memory_manager.add_step(task_id, request.message, answer)
        elif current_task:
            memory_manager.add_step(current_task["current_task_id"], request.message, answer)

        return {"response": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)