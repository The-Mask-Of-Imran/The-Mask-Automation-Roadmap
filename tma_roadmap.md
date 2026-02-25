

ভিশন, আর্কিটেকচার, কোর ক্যাপাবিলিটিস, রোডম্যাপ সামারি, এবং বর্তমান ইমপ্লিমেন্টেশন বিস্তারিত।
সিস্টেমের নাম এবং মূল দর্শন
নাম: The Mask Personal AI Core v1.0 (অথবা The Mask Automation Core System, যেমন Render এবং GitHub-এ উল্লেখিত)।
মূল দর্শন: এটি একটা হাইব্রিড, সেলফ-আপগ্রেডেবল, লং-টার্ম মেমরি-সম্পন্ন ব্যক্তিগত AI যা তোমার (অ্যাডমিনের) সবকিছু শেখে, ভুল থেকে শিখে, এবং নিজেকে নিজে আপগ্রেড করে। এটি তোমার PC অটোমেশনের কোর হিসেবে কাজ করবে – চ্যাট থেকে শুরু করে ফাইল ম্যানেজমেন্ট, CMD এক্সিকিউশন, TTS/STT, এবং ভবিষ্যতে হ্যাকিং/PC কন্ট্রোল পর্যন্ত। প্রথম প্রায়োরিটি: সেলফ-আপগ্রেড ক্যাপাবিলিটি, যাতে পরবর্তী ফিচারগুলো নিজে যোগ করতে পারে। এটি "বুদ্ধিমান" হবে মানে, এটি ভুল থেকে লার্ন করে দ্বিতীয়বার একই ভুল করবে না, এবং বছরের পর বছর টাস্ক মনে রাখবে।
আর্কিটেকচার (হাইব্রিড ডিজাইন)
সিস্টেমটি হাইব্রিড – ক্লাউড এবং লোকাল লেয়ারের কম্বিনেশন:
ক্লাউড লেয়ার (Render.com):
মডেল: 7B প্যারামিটার LLM (যেমন Llama বা অনুরূপ), Groq 70B ফলব্যাক সহ।
সুবিধা: ১-৫০টা API একসাথে চালানো, স্মার্ট সুইচিং (লিমিট শেষ হলে অটো সুইচ), কম খরচে স্কেলেবল।
ব্যবহার: লাইটওয়েট টাস্ক, চ্যাট, এবং ফাস্ট রেসপন্স।
লোকাল লেয়ার (তোমার PC):
মডেল: 14B প্যারামিটার LLM (Ollama দিয়ে রান)।
সুবিধা: ভারী কাজ (e.g., পার্সোনাল ফাইল অ্যানালাইসিস, লং-টার্ম মেমরি প্রসেসিং), প্রাইভেসি (ডাটা লোকাল থাকে), কোনো API লিমিট নেই।
ব্যবহার: সেনসিটিভ ডাটা, অফলাইন মোড।
স্মার্ট সুইচিং ইঞ্জিন:
কাজের ধরন (e.g., সিম্পল চ্যাট vs কম্প্লেক্স কোডিং), খরচ, স্পিড, API লিমিট, অ্যাভেলেবিলিটি অনুসারে অটো সিদ্ধান্ত নেয়।
উদাহরণ: API লিমিট শেষ হলে লোকালে সুইচ, অথবা সেনসিটিভ টাস্ক লোকালে।
ডাটা স্টোরেজ এবং সিঙ্ক: SQLite ডাটাবেস (প্রাইমারি) + JSON ব্যাকআপ। ক্লাউড এবং লোকালের মধ্যে অটো সিঙ্ক্রোনাইজেশন (মেমরি শেয়ার)।
সিকিউরিটি লেয়ার: অ্যাপ্রুভাল সিস্টেম (সেনসিটিভ অপারেশনের আগে ইউজার অনুমতি), এনক্রিপশন (সেনসিটিভ ডাটা), রুল-বেসড চেকস।
কোর ক্যাপাবিলিটিস (প্রথমে তৈরি করা ফিচারস)
সিস্টেমের মূল ক্ষমতা নিচেরগুলো, যা রোডম্যাপ অনুসারে ধাপে ধাপে যোগ হবে:
যেকোনো ফিচার অনুরোধ অ্যানালাইজ: ইউজারের কমান্ড (e.g., "TTS যোগ করো") অ্যানালাইজ করে কোড জেনারেট এবং আপগ্রেড।
সিস্টেম অ্যাক্সেস: অনুমতি সাপেক্ষে ফাইল তৈরি/এডিট/ডিলিট, CMD/PowerShell এক্সিকিউশন।
কোডিং দক্ষতা: এরর অটো ফিক্স, কোড জেনারেশন।
সেলফ-আপগ্রেড: নিজে কোড লিখে, টেস্ট করে, git push, Render রিস্টার্ট। ব্যাকআপ (zip + Google Drive), রোলব্যাক মেকানিজম সহ।
লং-টার্ম মেমরি: টাস্ক, ইন্টার্যাকশন, লার্নিং লেসন সেভ (বছরের পর বছর)। অটো সামারাইজেশন পুরনো মেমরির।
সেলফ-লার্নিং: ভুল থেকে লেসন সেভ, হাইব্রিড লার্নিং (লোকাল + Groq)।
অ্যাডভান্সড ফিচারস (পরবর্তীতে): TTS/STT (ভয়েস ইন্টারফেস), প্লাগিন সিস্টেম, ড্যাশবোর্ড UI (Streamlit), লং-টার্ম টাস্ক ম্যানেজমেন্ট।
এরর হ্যান্ডলিং এবং মনিটরিং: লগিং, অটো-লার্নিং, /status এন্ডপয়েন্ট (সব মডেলের স্টেট)।
রোডম্যাপ সামারি (৩টা ফেজ)
ফেজ ১: কোর ফাউন্ডেশন + হাইব্রিড + সিকিউরিটি (৪-৫ সপ্তাহ): লং-টার্ম মেমরি ফিক্স, হাইব্রিড সুইচিং, অ্যাপ্রুভাল সিস্টেম, লোকাল কানেকশন, সিঙ্ক্রোনাইজেশন। লক্ষ্য: স্থিতিশীল বেস।
ফেজ ২: সেলফ-আপগ্রেড ইঞ্জিন (৪-৬ সপ্তাহ): SelfUpgradeEngine, কোড ফ্লো, ব্যাকআপ/রোলব্যাক, অটো প্রস্তাব, টেস্টস। লক্ষ্য: নিজে আপগ্রেড করার ক্ষমতা।
ফেজ ৩: অ্যাডভান্সড লার্নিং + লং-টার্ম টাস্ক (৩-৪ সপ্তাহ): SelfLearningManager, টাস্ক স্টেট, হাইব্রিড লার্নিং, সামারাইজেশন, সিকিউরিটি অডিট, প্লাগিন বেস, ভয়েস ইন্টারফেস, ড্যাশবোর্ড, E2E টেস্ট, v1.0 রিলিজ। লক্ষ্য: সত্যিকারের বুদ্ধিমান সিস্টেম।
সময়কাল: মোট ১১-১৫ সপ্তাহ, ছোট টাস্কে ভাগ করা।
বর্তমান তৈরিকৃত ইনফরমেশন এবং স্ট্যাটাস
তোমার প্রদত্ত লিঙ্কগুলো অ্যানালাইজ করে (Render.com সার্ভিস এবং GitHub রেপো), বর্তমান সিস্টেমের অবস্থা নিচের মতো:
Render.com সার্ভিস:
Service ID: srv-d6309dcr85hc739uvh0g
Service Address: https://the-mask-automation-core.onrender.com
বর্তমান ফাংশনালিটি: সার্ভিসটি চালু আছে এবং একটা সিম্পল JSON রেসপন্স দেয়: {"message": "The Mask Core System চালু আছে! বাংলায় কথা বলতে পারি।"}। এটি কোর চ্যাট ফিচারের ইন্ডিকেটর – সিস্টেম অ্যাকটিভ এবং বাংলায় ইন্টার্যাকশন সমর্থন করে। কোনো অতিরিক্ত এন্ডপয়েন্ট (e.g., /status, /chat) দৃশ্যমান নয়, কিন্তু এটি একটা API-ভিত্তিক সার্ভিস বলে মনে হয় যা চ্যাট বা অটোমেশনের বেস। লং-টার্ম মেমরি বা অন্য ফিচারস এখনো পাবলিকলি এক্সপোজড নয় (connection refused এররের কথা মনে করে, এটি আংশিক)।
স্ট্যাটাস: ✅ কমপ্লিট (কোর চ্যাট লাইভ), কিন্তু লং-টার্ম মেমরি ❌ (এখনো ফিক্স নয়)।
GitHub রেপো:
URL: https://github.com/The-Mask-Of-Imran/The-Mask-Core-System
ডেসক্রিপশন: "My PC Automator Core System Backup" – এটি তোমার PC অটোমেটরের কোর সিস্টেমের ব্যাকআপ।
ফাইলস এবং স্ট্রাকচার:
app.py: মূল অ্যাপ্লিকেশন স্ক্রিপ্ট (সম্ভবত Flask/FastAPI-ভিত্তিক, চ্যাট লজিক এখানে)।
config.py
# config.py
import os

# সব গুরুত্বপূর্ণ সেটিংস এখানে রাখা হলো
RENDER_URL = os.getenv("RENDER_URL", "https://the-mask-automation-core.onrender.com")
STATUS_AUTH_KEY = os.getenv("STATUS_AUTH_KEY", "secret_key_123")          # এটা পরিবর্তন করতে পারো
APPROVAL_TIMEOUT = int(os.getenv("APPROVAL_TIMEOUT", "30"))               # সেকেন্ড
SYNC_INTERVAL = int(os.getenv("SYNC_INTERVAL", "5"))                       # মিনিট
PLUGIN_DIR = os.getenv("PLUGIN_DIR", "plugins/")
TTS_TEST_URL = os.getenv("TTS_TEST_URL", "http://localhost:5000/tts")
DASHBOARD_AUTH_PASSWORD = os.getenv("DASHBOARD_AUTH_PASSWORD", "secure_pass_2025")
STATE_SAVE_INTERVAL = int(os.getenv("STATE_SAVE_INTERVAL", "3600"))       # সেকেন্ড (১ ঘণ্টা)
LEARNING_PROMPT = os.getenv("LEARNING_PROMPT", "Analyze this error: {error}. What lesson should the system learn? Categorize as code_error/performance_issue/security/etc.")

config.json: কনফিগারেশন ফাইল (API কী, মডেল সেটিংস, ইত্যাদি)।
memory.json: লং-টার্ম মেমরি স্টোরেজ (JSON-ভিত্তিক ব্যাকআপ, কিন্তু এখনো অস্থিতিশীল)।
requirements.txt: ডিপেন্ডেন্সিস লিস্ট (Python লাইব্রেরী যেমন requests, ollama, ইত্যাদি)।
কমিটস: মোট ৬টা কমিট (সাম্প্রতিক পরিবর্তনগুলো সিস্টেমের বেসিক সেটআপ)।
README: নেই (পরে যোগ করা যাবে ডকুমেন্টেশনের জন্য)।
স্ট্যাটাস: রেপোটি ব্যাকআপ হিসেবে কাজ করছে, কিন্তু পুরো রোডম্যাপের শুধুমাত্র কোর চ্যাট কমপ্লিট। বাকি টাস্কস (e.g., মেমরি ফিক্স, সেলফ-আপগ্রেড) ইমপ্লিমেন্ট করা বাকি।
সামগ্রিক বর্তমান স্ট্যাটাস (মেসেজ থেকে):
✅ কোর চ্যাট (ফিচার ১) কমপ্লিট এবং লাইভ (Render-এ)।
❌ লং-টার্ম মেমরি (ফিচার ২) – Connection refused এরর, ফিক্স দরকার।
বাকি সব ❌ (e.g., সেলফ-আপগ্রেড, হাইব্রিড সুইচিং)।
সমস্যা: মেমরি সিস্টেম অস্থিতিশীল, যা সেলফ-আপগ্রেডের জন্য আবশ্যক।


টাস্ক ১: লং-টার্ম মেমরি ফিক্স (Connection Refused Error Resolution with SQLite and JSON Backup)
ওভারভিউ: এই টাস্কে লং-টার্ম মেমরি সিস্টেমকে স্থিতিশীল করো, যাতে কানেকশন রিফিউজড এরর না আসে। SQLite প্রাইমারি স্টোরেজ, JSON ব্যাকআপ। TaskMemoryManager ক্লাস আপডেট করো।
প্রয়োজনীয় প্রিপারেশন:
Python 3.10+ ইনস্টল।
লাইব্রেরী: sqlite3 (built-in), json (built-in), logging (built-in), tenacity (pip install tenacity==8.0.1)।
GitHub রেপো ক্লোন: git clone https://github.com/The-Mask-Of-Imran/The-Mask-Core-System।
ডিরেক্টরি স্ট্রাকচার: modules/memory/ ফোল্ডার তৈরি যদি না থাকে।
স্টেপ-বাই-স্টেপ ইমপ্লিমেন্টেশন:
modules/memory/TaskMemoryManager.py ফাইল তৈরি করো (যদি না থাকে)।
ক্লাস ডিফাইন: class TaskMemoryManager:।
init মেথড যোগ:
db_path = 'data/memory.db' (ফিক্সড পাথ, data/ ফোল্ডার তৈরি যদি না থাকে)।
try: self.conn = sqlite3.connect(db_path, check_same_thread=False) # WAL মোড।
self.cursor = self.conn.cursor()।
self.cursor.execute('''CREATE TABLE IF NOT EXISTS memories (id INTEGER PRIMARY KEY, task_id TEXT, content TEXT, timestamp DATETIME, category TEXT)''')।
self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON memories (timestamp)') # পারফরম্যান্স।
self.conn.commit()।
except sqlite3.OperationalError as e: logging.error(f"DB Error: {e}")।
রিট্রাই: from tenacity import retry, wait_exponential, stop_after_attempt।
@retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(3)) def retry_connect(): self.conn = sqlite3.connect(db_path)।
retry_connect()।
save_memory মেথড যোগ:
def save_memory(self, task_id: str, content: dict, category: str):।
from datetime import datetime।
timestamp = datetime.now()।
self.cursor.execute("INSERT INTO memories (task_id, content, timestamp, category) VALUES (?, ?, ?, ?)", (task_id, json.dumps(content), timestamp, category))।
self.conn.commit()।
ব্যাকআপ: def _backup_to_json(self): with open('data/memory_backup.json', 'w') as f: json.dump(self._get_all_memories(), f, indent=4)।
self._backup_to_json() # প্রত্যেক সেভের পর কল।
load_memory মেথড যোগ:
def load_memory(self, category: str = None):।
query = "SELECT * FROM memories" if not category else "SELECT * FROM memories WHERE category = ?"।
params = () if not category else (category,)।
self.cursor.execute(query, params)।
rows = self.cursor.fetchall()।
return [{ 'id': row[0], 'task_id': row[1], 'content': json.loads(row[2]), 'timestamp': row[3], 'category': row[4] } for row in rows]।
_get_all_memories মেথড (প্রাইভেট): একই লজিক load_memory(None) এর মতো।
close মেথড: def close(self): self.conn.close()।
app.py-এ ইন্টিগ্রেট: from modules.memory.TaskMemoryManager import TaskMemoryManager।
memory_manager = TaskMemoryManager() # গ্লোবাল।
চ্যাট হ্যান্ডলারে: memory_manager.save_memory(task_id='chat1', content={'user': 'input'}, category='interaction')।
ইন্টিগ্রেশন: এই ক্লাস app.py-এর মেইন চ্যাট লজিকে কল করো। পরবর্তী টাস্ক (e.g., ১০) এখানকার save/load মেথড ইউজ করে সিঙ্ক করো। কোনো চেঞ্জে db_path ফিক্সড রাখো।
টেস্টিং:
tests/memory_test.py তৈরি: import unittest।
class TestMemoryManager(unittest.TestCase): def test_save_load(self): manager = TaskMemoryManager(); manager.save_memory('test', {'key': 'value'}, 'test'); memories = manager.load_memory('test'); assert len(memories) == 1।
python -m unittest tests/memory_test.py রান করে চেক।
ফাইনাল আউটপুট: TaskMemoryManager.py ফাইল কমপ্লিট, মেমরি সেভ/লোড কাজ করে, এরর ফিক্সড। git commit -m "Task 1: Memory Fix"।

টাস্ক ২: হাইব্রিড সুইচিং ইঞ্জিন (ModelRouter Class)
ওভারভিউ: ModelRouter ক্লাস তৈরি করো যা ক্লাউড (7B/Groq 70B) এবং লোকাল (14B) মধ্যে সুইচ করে – টাস্ক ধরন, লিমিট অনুসারে।
প্রয়োজনীয় প্রিপারেশন:
টাস্ক ১ কমপ্লিট।
লাইব্রেরী: requests (pip install requests==2.28.1), ollama (pip install ollama==0.0.18), os, logging।
env vars: os.environ['GROQ_API_KEY'] = 'your_key'; os.environ['RENDER_URL'] = 'https://the-mask-automation-core.onrender.com'।
স্টেপ-বাই-স্টেপ ইমপ্লিমেন্টেশন:
modules/router/ModelRouter.py ফাইল তৈরি।
class ModelRouter:।
init মেথড: self.models = {'cloud_7b': {'url': os.environ['RENDER_URL'] + '/api/7b', 'type': 'cloud'}, 'groq_70b': {'url': 'https://api.groq.com/v1/models/70b', 'key': os.environ['GROQ_API_KEY'], 'type': 'cloud'}, 'local_14b': {'model': 'llama-14b', 'type': 'local'}}।
route_request মেথড: def route_request(self, task_type: str, data_size: int, urgency: str):।
if data_size > 1000 or task_type == 'sensitive': return 'local_14b' # লজিক ফিক্সড।
elif urgency == 'high': return 'groq_70b'।
else: return 'cloud_7b'।
logging.info(f"Routed to {model} for task {task_type}")।
generate_response মেথড: def generate_response(self, prompt: str, task_type: str, data_size: int = 0, urgency: str = 'normal'):।
model = self.route_request(task_type, data_size, urgency)।
if model == 'local_14b': import ollama; return ollama.generate(model=self.models[model]['model'], prompt=prompt)['response']।
elif model == 'groq_70b': headers = {'Authorization': f"Bearer {self.models[model]['key']}"}; response = requests.post(self.models[model]['url'], json={'prompt': prompt}, headers=headers); return response.json()['response']।
else: response = requests.post(self.models[model]['url'], json={'prompt': prompt}); return response.json()['response']।
app.py-এ ইন্টিগ্রেট: from modules.router.ModelRouter import ModelRouter।
router = ModelRouter()।
চ্যাট হ্যান্ডলারে: response = router.generate_response(user_input, task_type='chat')।
from modules.memory.TaskMemoryManager import TaskMemoryManager; memory_manager = TaskMemoryManager(); memory_manager.save_memory('route', {'model': model}, 'routing') # টাস্ক ১ সিঙ্ক।
ইন্টিগ্রেশন: টাস্ক ১-এর TaskMemoryManager ইমপোর্ট করে সেভ করো। পরবর্তী টাস্ক (e.g., ৩) এখানকার route_request extend করো। env vars ফিক্সড রাখো।
টেস্টিং:
tests/router_test.py: class TestModelRouter(unittest.TestCase): def test_route(self): router = ModelRouter(); assert router.route_request('sensitive', 2000, 'low') == 'local_14b'।
python -m unittest tests/router_test.py।
ফাইনাল আউটপুট: ModelRouter.py কমপ্লিট, সুইচিং কাজ করে। git commit -m "Task 2: Hybrid Switching"।

টাস্ক ৩: API লিমিট মনিটরিং + স্মার্ট সুইচিং লজিক
ওভারভিউ: API লিমিট মনিটর করো এবং লিমিট শেষ হলে অটো সুইচ। ModelRouter-এ ইন্টিগ্রেট।
প্রয়োজনীয় প্রিপারেশন:
টাস্ক ২ কমপ্লিট।
লাইব্রেরী: requests, collections (built-in), threading (built-in)।
env vars: os.environ['API_LIMIT'] = '100' # per minute ফিক্সড।
স্টেপ-বাই-স্টেপ ইমপ্লিমেন্টেশন:
modules/router/ModelRouter.py আপডেট (টাস্ক ২-এর ফাইল)।
LimitMonitor ক্লাস যোগ: class LimitMonitor:।
init: from collections import deque; self.calls = deque(); self.limit = int(os.environ['API_LIMIT']); self.threshold = 0.1 * self.limit।
track_call মেথড: def track_call(self): self.calls.append(time.time()); while self.calls and self.calls[0] < time.time() - 60: self.calls.popleft()।
is_limit_exceeded মেথড: def is_limit_exceeded(self): return len(self.calls) >= self.limit।
ModelRouter-এ ইন্টিগ্রেট: self.limit_monitor = LimitMonitor() # init-এ।
generate_response-এ: self.limit_monitor.track_call()।
route_request-এ: if model.startswith('cloud') and self.limit_monitor.is_limit_exceeded(): return 'local_14b' # লজিক ফিক্সড।
Background thread: import threading; def monitor_loop(self): while True: time.sleep(60); logging.info("Limits reset") # ফিক্সড লজিক।
init-এ: threading.Thread(target=self.monitor_loop).start()।
app.py আপডেট: router.generate_response() কল অ্যাডজাস্ট করো, টাস্ক ১-এর memory_manager.save_memory('limit', {'exceeded': is_exceeded}, 'monitoring')।
ইন্টিগ্রেশন: টাস্ক ২-এর route_request extend। টাস্ক ১-এর save_memory কল। env 'API_LIMIT' ফিক্সড।
টেস্টিং:
tests/limit_test.py: def test_exceeded(self): monitor = LimitMonitor(); for _ in range(101): monitor.track_call(); assert monitor.is_limit_exceeded()।
python -m unittest।
ফাইনাল আউটপুট: ModelRouter.py আপডেট, লিমিট মনিটরিং কাজ করে। git commit -m "Task 3: API Limit Monitoring"।

#### **টাস্ক ৪: Approval System (ApprovalManager - First Version)**

**ওভারভিউ**: এই টাস্কে ApprovalManager ক্লাস তৈরি করো যা সেনসিটিভ অপারেশনের আগে ইউজার অ্যাপ্রুভাল নেবে। প্রথম ভার্সনে কনসোল প্রম্পট (yes/no) এবং লগিং। রুল-বেসড অটো-অ্যাপ্রুভ লো-রিস্ক অ্যাকশনের জন্য।

**প্রয়োজনীয় প্রিপারেশন**:
- টাস্ক ৩ কমপ্লিট (ModelRouter থেকে সেনসিটিভ টাস্ক চেক)।
- লাইব্রেরী: flask (pip install flask==2.2.2, API UI-এর জন্য), logging (built-in), input() (কনসোল)।
- GitHub রেপো ক্লোন: git clone https://github.com/The-Mask-Of-Imran/The-Mask-Core-System।
- ডিরেক্টরি স্ট্রাকচার: modules/approval/ ফোল্ডার তৈরি যদি না থাকে। env vars: os.environ['APPROVAL_TIMEOUT'] = '30' # সেকেন্ড ফিক্সড।

**স্টেপ-বাই-স্টেপ ইমপ্লিমেন্টেশন**:
1. modules/approval/ApprovalManager.py ফাইল তৈরি।
2. class ApprovalManager:।
3. __init__ মেথড: self.rules = {'low_risk': ['read_file'], 'high_risk': ['delete_file', 'upgrade_code']} # ফিক্সড রুলস।
   - from config import APPROVAL_TIMEOUT
      self.timeout = APPROVAL_TIMEOUT
4. request_approval মেথড: def request_approval(self, action: str, description: str):।
   - if action in self.rules['low_risk']: return True # অটো-অ্যাপ্রুভ লজিক।
   - prompt = f"Approve {action}? {description} (Y/N): "।
   - import time; start = time.time()।
   - while time.time() - start < self.timeout: user_input = input(prompt).strip().upper(); if user_input == 'Y': logging.info(f"Approved: {action}"); return True; elif user_input == 'N': logging.info(f"Denied: {action}"); return False।
   - logging.warning(f"Timeout for {action}"); return False # টাইমআউট লজিক।
5. API UI: from flask import Flask, request; app = Flask(__name__); @app.route('/approve', methods=['POST']): def approve(): data = request.json; approved = self.request_approval(data['action'], data['description']); return {'approved': approved}।
   - if __name__ == '__main__': app.run(port=5001) # ফিক্সড পোর্ট।
6. app.py-এ ইন্টিগ্রেট: from modules.approval.ApprovalManager import ApprovalManager।
   - approval_manager = ApprovalManager()।
   - সেনসিটিভ কলে: if approval_manager.request_approval('delete', 'Delete file X'): # execute।
   - from modules.memory.TaskMemoryManager import TaskMemoryManager; memory_manager = TaskMemoryManager(); memory_manager.save_memory('approval', {'action': action, 'approved': approved}, 'security') # টাস্ক ১ সিঙ্ক।
   - from modules.router.ModelRouter import ModelRouter; router = ModelRouter(); if 'sensitive' in task_type: approved = approval_manager.request_approval(...) # টাস্ক ২-৩ সিঙ্ক।

**ইন্টিগ্রেশন**: টাস্ক ১-এর TaskMemoryManager ইমপোর্ট করে সেভ করো। টাস্ক ২-৩-এর ModelRouter-এ sensitive চেকে request_approval কল। পরবর্তী টাস্ক (e.g., ৫) এখানকার request_approval ইউজ করে। env 'APPROVAL_TIMEOUT' ফিক্সড রাখো।

**টেস্টিং**:
1. tests/approval_test.py: class TestApprovalManager(unittest.TestCase): def test_auto_approve(self): manager = ApprovalManager(); assert manager.request_approval('read_file', 'Test') == True।
   - def test_prompt(self): # মক ইনপুট দিয়ে টেস্ট, e.g., monkeypatch input।
2. python -m unittest tests/approval_test.py।

**ফাইনাল আউটপুট**: ApprovalManager.py কমপ্লিট, অ্যাপ্রুভাল প্রম্পট কাজ করে। git commit -m "Task 4: Approval System"।

---

#### **টাস্ক ৫: Basic File + CMD Execution Module (Permission-Based)**

**ওভারভিউ**: ExecutionModule ক্লাস তৈরি করো যা ফাইল অপারেশন এবং CMD এক্সিকিউট করবে, কিন্তু অ্যাপ্রুভাল সাপেক্ষে। স্যান্ডবক্সড (প্রোজেক্ট ফোল্ডারে লিমিটেড)।

**প্রয়োজনীয় প্রিপারেশন**:
- টাস্ক ৪ কমপ্লিট (ApprovalManager থেকে চেক)।
- লাইব্রেরী: subprocess (built-in), os (built-in), shutil (built-in), logging।
- env vars: os.environ['SANDBOX_DIR'] = 'project_folder' # ফিক্সড ডির।
- ডিরেক্টরি: project_folder/ তৈরি।

**স্টেপ-বাই-স্টেপ ইমপ্লিমেন্টেশন**:
1. modules/execution/ExecutionModule.py ফাইল তৈরি।
2. class ExecutionModule:।
3. __init__ মেথড: self.sandbox_dir = os.environ['SANDBOX_DIR']; if not os.path.exists(self.sandbox_dir): os.makedirs(self.sandbox_dir)।
4. file_op মেথড: def file_op(self, op: str, file_path: str, content: str = None):।
   - full_path = os.path.join(from config import SANDBOX_DIR   # (যদি config-এ যোগ করা না থাকে, তাহলে config.py-তে যোগ করো: SANDBOX_DIR = os.getenv("SANDBOX_DIR", "project_folder"))
self.sandbox_dir = SANDBOX_DIR) # স্যান্ডবক্স লজিক।
   - from modules.approval.ApprovalManager import ApprovalManager; approval_manager = ApprovalManager()।
   - if not approval_manager.request_approval(op, f"{op} on {file_path}"): return "Denied"।
   - if op == 'create': with open(full_path, 'w') as f: f.write(content); logging.info(f"Created {full_path}")।
   - elif op == 'delete': os.remove(full_path); logging.info(f"Deleted {full_path}")।
   - elif op == 'edit': with open(full_path, 'a') as f: f.write(content); logging.info(f"Edited {full_path}")।
   - except Exception as e: logging.error(f"File op error: {e}"); return str(e)।
5. execute_cmd মেথড: def execute_cmd(self, cmd: str):।
   - approval_manager = ApprovalManager()।
   - if not approval_manager.request_approval('cmd', f"Execute {cmd}"): return "Denied"।
   - import subprocess; try: output = subprocess.check_output(cmd, shell=True, cwd=self.sandbox_dir); return output.decode('utf-8') except Exception as e: logging.error(f"CMD error: {e}"); return str(e) # আউটপুট ক্যাপচার।
6. app.py-এ ইন্টিগ্রেট: from modules.execution.ExecutionModule import ExecutionModule।
   - executor = ExecutionModule()।
   - চ্যাট হ্যান্ডলারে: if 'file' in command: result = executor.file_op(...);।
   - from modules.memory.TaskMemoryManager import TaskMemoryManager; memory_manager = TaskMemoryManager(); memory_manager.save_memory('exec', {'cmd': cmd, 'output': output}, 'execution') # টাস্ক ১ সিঙ্ক।
   - from modules.router.ModelRouter import ModelRouter; router = ModelRouter(); response = router.generate_response(f"Exec result: {result}", 'execution') # টাস্ক ২-৩ সিঙ্ক।

**ইন্টিগ্রেশন**: টাস্ক ৪-এর ApprovalManager ইমপোর্ট করে চেক করো। টাস্ক ১-এর save_memory কল। টাস্ক ২-৩-এর ModelRouter-এ execution টাস্ক রাউট। পরবর্তী টাস্ক (e.g., ৬) এখানকার execute_cmd ইউজ করে। env 'SANDBOX_DIR' ফিক্সড রাখো।

**টেস্টিং**:
1. tests/execution_test.py: class TestExecutionModule(unittest.TestCase): def test_file_op(self): executor = ExecutionModule(); assert executor.file_op('create', 'test.txt', 'content') != "Denied" # মক অ্যাপ্রুভাল।
2. python -m unittest tests/execution_test.py।

**ফাইনাল আউটপুট**: ExecutionModule.py কমপ্লিট, ফাইল/CMD এক্সিকিউশন অ্যাপ্রুভাল সহ কাজ করে। git commit -m "Task 5: File CMD Execution"।

---

#### **টাস্ক ৬: Local 14B Model Connection (Ollama Integration)**

**ওভারভিউ**: LocalModelConnector ক্লাস তৈরি করো যা Ollama দিয়ে লোকাল 14B মডেল কানেক্ট করবে। অটো-স্টার্ট, রিকানেক্ট, স্টেট মনিটর। ModelRouter-এ ইন্টিগ্রেট।

**প্রয়োজনীয় প্রিপারেশন**:
- টাস্ক ৫ কমপ্লিট (ExecutionModule থেকে CMD স্টার্ট)।
- লাইব্রেরী: ollama (pip install ollama==0.0.18), requests (HTTP চেক), subprocess (স্টার্ট), logging।
- env vars: os.environ['OLLAMA_MODEL'] = 'llama-14b'; os.environ['OLLAMA_PORT'] = '11434' # ফিক্সড।
- Ollama ইনস্টল এবং মডেল ডাউনলোড: ollama pull llama-14b।

**স্টেপ-বাই-স্টেপ ইমপ্লিমেন্টেশন**:
1. modules/local/LocalModelConnector.py ফাইল তৈরি।
2. class LocalModelConnector:।
3. __init__ মেথড: self.model = os.environ['OLLAMA_MODEL']; self.port = os.environ['OLLAMA_PORT']; self._check_and_start() # অটো-স্টার্ট।
4. _check_and_start মেথড: def _check_and_start(self):।
   - try: response = requests.get(f"http://localhost:{self.port}/health"); if response.status_code != 200: raise ConnectionError।
   - except ConnectionError: import subprocess; subprocess.Popen(['ollama', 'serve']) # ফিক্সড কমান্ড।
   - from tenacity import retry, stop_after_attempt; @retry(stop=stop_after_attempt(3)); def connect(): requests.get(f"http://localhost:{self.port}/health")।
   - connect(); logging.info("Ollama connected")।
5. generate_response মেথড: def generate_response(self, prompt: str): return ollama.generate(model=self.model, prompt=prompt)['response'] # ফিক্সড।
6. get_state মেথড: def get_state(self): try: requests.get(f"http://localhost:{self.port}/health"); return 'running' except: return 'idle'।
7. app.py-এ ইন্টিগ্রেট: from modules.local.LocalModelConnector import LocalModelConnector।
   - local_connector = LocalModelConnector()।
   - from modules.router.ModelRouter import ModelRouter; router = ModelRouter(); if model == 'local_14b': response = local_connector.generate_response(prompt) # টাস্ক ২-৩ সিঙ্ক।
   - from modules.approval.ApprovalManager import ApprovalManager; approval_manager = ApprovalManager(); if approval_manager.request_approval('local_exec', 'Run local model'): ... # টাস্ক ৪ সিঙ্ক।
   - from modules.execution.ExecutionModule import ExecutionModule; executor = ExecutionModule(); executor.execute_cmd('ollama pull llama-14b') if not connected # টাস্ক ৫ সিঙ্ক।
   - from modules.memory.TaskMemoryManager import TaskMemoryManager; memory_manager = TaskMemoryManager(); memory_manager.save_memory('local', {'state': state}, 'model') # টাস্ক ১ সিঙ্ক।

**ইন্টিগ্রেশন**: টাস্ক ৪-এর ApprovalManager ইমপোর্ট করে চেক করো। টাস্ক ৫-এর execute_cmd ইউজ করে Ollama স্টার্ট। টাস্ক ১-এর save_memory কল। টাস্ক ২-৩-এর ModelRouter-এ local_14b রাউটে generate_response কল। পরবর্তী টাস্ক (e.g., ৭) এখানকার get_state ইউজ করে। env 'OLLAMA_MODEL' ফিক্সড রাখো।

**টেস্টিং**:
1. tests/local_test.py: class TestLocalModelConnector(unittest.TestCase): def test_connect(self): connector = LocalModelConnector(); assert connector.get_state() == 'running'।
   - def test_response(self): assert 'test' in connector.generate_response('Echo test')।
2. python -m unittest tests/local_test.py।

**ফাইনাল আউটপুট**: LocalModelConnector.py কমপ্লিট, লোকাল মডেল কানেকশন কাজ করে। git commit -m "Task 6: Local Model Connection"।

#### **টাস্ক ৭: Smart Switching Test (Cloud vs Local)**

**ওভারভিউ**: এই টাস্কে TestSuite ক্লাস তৈরি করো যা হাইব্রিড সুইচিং টেস্ট করবে। টেস্ট কেস: হাই-লোড, লিমিট-এক্সসিড, অফলাইন সিনারিও। রিপোর্ট জেনারেট: JSON সহ সাকসেস রেট, টাইমিং, ফেল কজ।

**প্রয়োজনীয় প্রিপারেশন**:
- টাস্ক ৬ কমপ্লিট (LocalModelConnector থেকে লোকাল চেক)।
- লাইব্রেরী: pytest (pip install pytest==7.2.0), unittest.mock (built-in), time (built-in), json (built-in), concurrent.futures (built-in), logging।
- GitHub রেপো ক্লোন: git clone https://github.com/The-Mask-Of-Imran/The-Mask-Core-System।
- ডিরেক্টরি স্ট্রাকচার: tests/ ফোল্ডার তৈরি যদি না থাকে। env vars: os.environ['TEST_REPORT_PATH'] = 'reports/switching_report.json' # ফিক্সড।

**স্টেপ-বাই-স্টেপ ইমপ্লিমেন্টেশন**:
1. tests/switching_test.py ফাইল তৈরি।
2. class TestSuite:।
3. __init__ মেথড: self.router = ModelRouter() # from modules.router.ModelRouter import ModelRouter।
   - self.report = {'success_rate': 0, 'tests': []} # ফিক্সড স্ট্রাকচার।
4. run_tests মেথড: def run_tests(self):।
   - tests = ['high_load', 'limit_exceed', 'offline_local'] # ফিক্সড কেসস।
   - import concurrent.futures; with concurrent.futures.ThreadPoolExecutor() as executor: results = list(executor.map(self._run_single_test, tests))।
   - success_count = sum(1 for r in results if r['success'])।
   - self.report['success_rate'] = (success_count / len(tests)) * 100।
   - self.report['tests'] = results।
   - with open(os.environ['TEST_REPORT_PATH'], 'w') as f: json.dump(self.report, f, indent=4); logging.info("Report generated")।
5. _run_single_test মেথড: def _run_single_test(self, test_type: str):।
   - import time; start = time.time()。
   - try:।
     - if test_type == 'high_load': for _ in range(10): self.router.route_request('chat', 500, 'high'); success = True # কনকারেন্ট সিমুলেট।
     - elif test_type == 'limit_exceed': from unittest.mock import patch; with patch('modules.router.ModelRouter.limit_monitor.is_limit_exceeded', return_value=True): model = self.router.route_request('chat', 100, 'normal'); success = model == 'local_14b'।
     - elif test_type == 'offline_local': with patch('requests.get', side_effect=ConnectionError): model = self.router.route_request('sensitive', 2000, 'low'); success = model != 'local_14b' # অফলাইন সিমুলেট।
     - error_type = None।
   - except Exception as e: success = False; error_type = str(e)।
   - time_taken = time.time() - start।
   - return {'type': test_type, 'success': success, 'time_taken': time_taken, 'error_type': error_type}।
6. app.py-এ ইন্টিগ্রেট: from tests.switching_test import TestSuite।
   - if __name__ == '__main__': suite = TestSuite(); suite.run_tests() # ম্যানুয়াল রান।
   - from modules.memory.TaskMemoryManager import TaskMemoryManager; memory_manager = TaskMemoryManager(); memory_manager.save_memory('test', self.report, 'testing') # টাস্ক ১ সিঙ্ক।
   - from modules.local.LocalModelConnector import LocalModelConnector; local_connector = LocalModelConnector(); if local_connector.get_state() == 'idle': logging.warning("Local offline for test") # টাস্ক ৬ সিঙ্ক।
   - from modules.approval.ApprovalManager import ApprovalManager; approval_manager = ApprovalManager(); if approval_manager.request_approval('test_run', 'Run switching tests'): suite.run_tests() # টাস্ক ৪ সিঙ্ক।
   - from modules.execution.ExecutionModule import ExecutionModule; executor = ExecutionModule(); executor.execute_cmd('pytest tests/switching_test.py') # টাস্ক ৫ সিঙ্ক।

**ইন্টিগ্রেশন**: টাস্ক ৬-এর LocalModelConnector ইমপোর্ট করে স্টেট চেক। টাস্ক ৪-এর ApprovalManager ইমপোর্ট করে টেস্ট রান অ্যাপ্রুভ। টাস্ক ৫-এর execute_cmd ইউজ করে pytest রান। টাস্ক ১-এর save_memory কল। টাস্ক ২-৩-এর ModelRouter টেস্ট। পরবর্তী টাস্ক (e.g., ৮) এখানকার report ইউজ করে। env 'TEST_REPORT_PATH' ফিক্সড রাখো।

**টেস্টিং**:
1. tests/switching_test.py-এ সেল্ফ-টেস্ট: if __name__ == '__main__': suite = TestSuite(); suite.run_tests(); assert suite.report['success_rate'] > 90।
2. python tests/switching_test.py রান করে চেক।

**ফাইনাল আউটপুট**: switching_test.py কমপ্লিট, টেস্ট রান করে রিপোর্ট জেনারেট। git commit -m "Task 7: Smart Switching Test"।

---

#### **টাস্ক ৮: Error Logging + Auto-Learning Basic Framework**

**ওভারভিউ**: ErrorLogger ক্লাস তৈরি করো যা এরর লগ করে এবং অটো-লার্নিং করে লেসন সেভ (মেমরিতে)। প্যাটার্ন ম্যাচ দিয়ে ফিউচার এরর অ্যাভয়েড।

**প্রয়োজনীয় প্রিপারেশন**:
- টাস্ক ৭ কমপ্লিট (TestSuite থেকে এরর সিমুলেট)।
- লাইব্রেরী: logging (built-in), traceback (built-in), re (built-in), json (built-in)।
- env vars: os.environ['ERROR_PATTERNS'] = json.dumps({'connection_refused': r'Connection refused', 'limit_exceeded': r'API limit'}) # ফিক্সড ডিক্ট।

**স্টেপ-বাই-স্টেপ ইমপ্লিমেন্টেশন**:
1. modules/error/ErrorLogger.py ফাইল তৈরি।
2. class ErrorLogger:।
3. __init__ মেথড: self.patterns = json.loads(os.environ['ERROR_PATTERNS']) # ফিক্সড প্যাটার্নস।
   - logging.basicConfig(filename='logs/errors.log', level=logging.ERROR) # ফিক্সড লগ ফাইল।
4. log_error মেথড: def log_error(self, e: Exception, context: str):।
   - import traceback; stack = traceback.format_exc()。
   - logging.error(f"Error: {str(e)}, Context: {context}, Stack: {stack}")।
   - self._analyze_and_learn(str(e))।
5. _analyze_and_learn মেথড: def _analyze_and_learn(self, error_msg: str):।
   - import re; for pattern_name, regex in self.patterns.items(): if re.search(regex, error_msg): lesson = f"Avoid {pattern_name} by checking before action"।
     - from modules.memory.TaskMemoryManager import TaskMemoryManager; memory_manager = TaskMemoryManager(); memory_manager.save_memory('error_lesson', {'lesson': lesson, 'error': error_msg}, 'learning') # টাস্ক ১ সিঙ্ক।
     - logging.info(f"Learned: {lesson}")।
6. check_learned_lessons মেথড: def check_learned_lessons(self, action: str):।
   - memory_manager = TaskMemoryManager(); lessons = memory_manager.load_memory('learning')।
   - for lesson in lessons: if action in lesson['content']['lesson']: return False # অ্যাভয়েড লজিক।
   - return True。
7. গ্লোবাল হ্যান্ডলার: import sys; sys.excepthook = lambda type, value, tb: self.log_error(value, 'Global') # ফিক্সড।
8. app.py-এ ইন্টিগ্রেট: from modules.error.ErrorLogger import ErrorLogger।
   - error_logger = ErrorLogger()।
   - try: ... except Exception as e: error_logger.log_error(e, 'Chat handler')।
   - before action: if not error_logger.check_learned_lessons('route'): return "Avoided known error"।
   - from modules.router.ModelRouter import ModelRouter; router = ModelRouter(); # টাস্ক ২-৩ সিঙ্ক, এরর লগ।
   - from modules.approval.ApprovalManager import ApprovalManager; approval_manager = ApprovalManager(); if approval_manager.request_approval('learn', 'Apply learned lesson'): ... # টাস্ক ৪ সিঙ্ক।
   - from modules.execution.ExecutionModule import ExecutionModule; executor = ExecutionModule(); executor.execute_cmd('echo test') # টাস্ক ৫ সিঙ্ক, এরর টেস্ট।
   - from modules.local.LocalModelConnector import LocalModelConnector; local_connector = LocalModelConnector(); # টাস্ক ৬ সিঙ্ক, কানেকশন এরর লগ।

**ইন্টিগ্রেশন**: টাস্ক ৭-এর TestSuite-এ log_error কল করে টেস্ট। টাস্ক ৪-এর ApprovalManager ইমপোর্ট করে লার্ন অ্যাপ্রুভ। টাস্ক ৫-এর execute_cmd-এ এরর হ্যান্ডেল। টাস্ক ৬-এর LocalModelConnector-এ কানেকশন এরর লগ। টাস্ক ১-এর save_memory কল। টাস্ক ২-৩-এর ModelRouter-এ এরর লগ। পরবর্তী টাস্ক (e.g., ৯) এখানকার check_learned_lessons ইউজ করে। env 'ERROR_PATTERNS' ফিক্সড রাখো।

**টেস্টিং**:
1. tests/error_test.py: class TestErrorLogger(unittest.TestCase): def test_log_and_learn(self): logger = ErrorLogger(); try: raise ConnectionError("Connection refused") except Exception as e: logger.log_error(e, 'Test'); lessons = memory_manager.load_memory('learning'); assert len(lessons) > 0।
2. python -m unittest tests/error_test.py।

**ফাইনাল আউটপুট**: ErrorLogger.py কমপ্লিট, এরর লগিং এবং লার্নিং কাজ করে। git commit -m "Task 8: Error Logging Auto-Learning"।

---

#### **টাস্ক ৯: /status Endpoint Upgrade (Show All Models' State)**

**ওভারভিউ**: /status এন্ডপয়েন্ট আপগ্রেড করো যা সব মডেলের স্টেট (running/idle/down, stats) দেখাবে JSON-এ। হেলথ চেক পিং দিয়ে।

**প্রয়োজনীয় প্রিপারেশন**:
- টাস্ক ৮ কমপ্লিট (ErrorLogger থেকে লগিং)।
- লাইব্রেরী: flask (pip install flask==2.2.2, API), requests (হেলথ), json (response), time (uptime)।
- env vars: os.environ['STATUS_AUTH_KEY'] = 'secret_key' # ফিক্সড অথ।

**স্টেপ-বাই-স্টেপ ইমপ্লিমেন্টেশন**:
1. app.py (মেইন ফাইল) আপডেট।
2. from flask import Flask, jsonify, request; app = Flask(__name__) # যদি না থাকে।
3. @app.route('/status', methods=['GET']) def get_status():।
   - if request.from config import STATUS_AUTH_KEY
headers = {'Auth-Key': STATUS_AUTH_KEY}}), 401 # সিকিউরিটি লজিক।
   - from modules.router.ModelRouter import ModelRouter; router = ModelRouter()।
   - from modules.local.LocalModelConnector import LocalModelConnector; local_connector = LocalModelConnector()।
   - import time; uptime = time.time() - start_time # গ্লোবাল start_time = time.time() __init__-এ।
   - status = {'models': {।
     - 'cloud_7b': {'status': 'running' if requests.get(router.models['cloud_7b']['url'] + '/health').status_code == 200 else 'down', 'uptime': uptime, 'requests': len(router.limit_monitor.calls) if hasattr(router, 'limit_monitor') else 0},।
     - 'groq_70b': {'status': 'running' if requests.get(router.models['groq_70b']['url'], headers={'Authorization': f"Bearer {router.models['groq_70b']['key']}"}, timeout=5).status_code == 200 else 'down', 'uptime': uptime, 'limits': router.limit_monitor.limit - len(router.limit_monitor.calls)},।
     - 'local_14b': {'status': local_connector.get_state(), 'uptime': uptime, 'health': 'ok' if local_connector.get_state() == 'running' else 'error'}},।
   - 'system_health': 'ok' if all(s['status'] == 'running' for s in status['models'].values()) else 'partial' }।
   - logging.info("Status requested"); return jsonify(status) # ফিক্সড response।
4. হেলথ হেল্পার: def _ping_model(url, headers=None): try: return requests.get(url, headers=headers, timeout=5).status_code == 200 except: return False # ফিক্সড।
5. app.py-এ গ্লোবাল: from modules.error.ErrorLogger import ErrorLogger; error_logger = ErrorLogger(); try: ... except: error_logger.log_error(...) # টাস্ক ৮ সিঙ্ক।
   - from modules.memory.TaskMemoryManager import TaskMemoryManager; memory_manager = TaskMemoryManager(); memory_manager.save_memory('status', status, 'monitoring') # টাস্ক ১ সিঙ্ক।
   - from modules.approval.ApprovalManager import ApprovalManager; approval_manager = ApprovalManager(); if approval_manager.request_approval('status_access', 'Access /status'): ... # টাস্ক ৪ সিঙ্ক।
   - from modules.execution.ExecutionModule import ExecutionModule; executor = ExecutionModule(); executor.execute_cmd('curl /status') # টাস্ক ৫ সিঙ্ক, টেস্ট।

**ইন্টিগ্রেশন**: টাস্ক ৮-এর ErrorLogger ইমপোর্ট করে লগ। টাস্ক ৪-এর ApprovalManager ইমপোর্ট করে অ্যাক্সেস অ্যাপ্রুভ। টাস্ক ৫-এর execute_cmd ইউজ করে curl টেস্ট। টাস্ক ১-এর save_memory কল। টাস্ক ২-৩-এর ModelRouter, টাস্ক ৬-এর LocalModelConnector ইউজ করে স্টেট। পরবর্তী টাস্ক (e.g., ১০) এখানকার /status কল করে। env 'STATUS_AUTH_KEY' ফিক্সড রাখো।

**টেস্টিং**:
1. tests/status_test.py: class TestStatus(unittest.TestCase): def test_endpoint(self): response = requests.get('http://localhost:5000/status', headers={'Auth-Key': 'secret_key'}); assert response.json()['system_health'] == 'ok'।
2. python -m unittest tests/status_test.py।

**ফাইনাল আউটপুট**: app.py আপডেট, /status এন্ডপয়েন্ট কাজ করে। git commit -m "Task 9: Status Endpoint Upgrade"।

#### **টাস্ক ১০: Render + Local Synchronization (Memory Share)**

**ওভারভিউ**: এই টাস্কে SyncEngine ক্লাস তৈরি করো যা ক্লাউড (Render) এবং লোকাল মধ্যে মেমরি সিঙ্ক করবে। বিডিরেকশনাল, কনফ্লিক্ট রেজল্যুশন (টাইমস্ট্যাম্প-বেসড), অফলাইন কিউ। পিরিয়ডিক সিঙ্ক (৫ মিনিট)।

**প্রয়োজনীয় প্রিপারেশন**:
- টাস্ক ৯ কমপ্লিট (/status থেকে হেলথ চেক)।
- লাইব্রেরী: requests (pip install requests==2.28.1), sqlite3 (built-in), json (built-in), schedule (pip install schedule==1.1.0), threading (built-in), logging।
- env vars: from config import SYNC_INTERVAL
self.interval = SYNC_INTERVAL
 = '5' # মিনিট ফিক্সড; os.environ['LOCAL_SYNC_URL'] = 'http://localhost:5001/sync' # ফিক্সড লোকাল এন্ডপয়েন্ট।
- Render-এ /sync এন্ডপয়েন্ট যোগ (app.py-এ)।

**স্টেপ-বাই-স্টেপ ইমপ্লিমেন্টেশন**:
1. modules/sync/SyncEngine.py ফাইল তৈরি।
2. class SyncEngine:।
3. __init__ মেথড: self.local_url = os.environ['LOCAL_SYNC_URL']; self.render_url = os.environ['RENDER_URL'] + '/sync'; self.interval = int(os.environ['SYNC_INTERVAL'])।
   - self.offline_queue = [] # লিস্ট ফিক্সড অফলাইন কিউ।
   - import threading; threading.Thread(target=self._periodic_sync, daemon=True).start() # ব্যাকগ্রাউন্ড।
4. sync_memories মেথড: def sync_memories(self, direction: str = 'bidirectional'):।
   - from modules.memory.TaskMemoryManager import TaskMemoryManager; memory_manager = TaskMemoryManager()।
   - local_memories = memory_manager.load_memory() # লোকাল থেকে।
   - try: response = requests.get(self.render_url); render_memories = response.json() if response.status_code == 200 else [] except: self._queue_for_later(local_memories); return "Offline sync queued" # অফলাইন লজিক।
   - merged = self._resolve_conflicts(local_memories, render_memories) # কনফ্লিক্ট লজিক।
   - if direction in ['to_render', 'bidirectional']: requests.post(self.render_url, json=merged)।
   - if direction in ['to_local', 'bidirectional']: for mem in merged: memory_manager.save_memory(mem['task_id'], mem['content'], mem['category']) # বিডিরেকশনাল।
   - logging.info("Sync completed"); return "Success"।
5. _resolve_conflicts মেথড: def _resolve_conflicts(self, local: list, render: list):।
   - from datetime import datetime; mem_dict = {}।
   - for mem in local + render: key = mem['task_id']; if key not in mem_dict or datetime.strptime(mem['timestamp'], '%Y-%m-%d %H:%M:%S.%f') > datetime.strptime(mem_dict[key]['timestamp'], '%Y-%m-%d %H:%M:%S.%f'): mem_dict[key] = mem # টাইমস্ট্যাম্প-বেসড, লেটেস্ট উইনস ফিক্সড।
   - return list(mem_dict.values())।
6. _queue_for_later মেথড: def _queue_for_later(self, data: list): self.offline_queue.append(data); logging.warning("Queued for sync") # অফলাইন কিউ।
7. _periodic_sync মেথড: def _periodic_sync(self): import schedule, time; schedule.every(self.interval).minutes.do(self.sync_memories); while True: schedule.run_pending(); time.sleep(1) # পিরিয়ডিক লজিক।
8. লোকাল /sync এন্ডপয়েন্ট: app.py-এ @app.route('/sync', methods=['GET', 'POST']): def sync(): if request.method == 'POST': data = request.json; memory_manager.save_memory(...) for d in data; return 'OK'; else: return jsonify(memory_manager.load_memory()) # ফিক্সড।
9. app.py-এ ইন্টিগ্রেট: from modules.sync.SyncEngine import SyncEngine।
   - sync_engine = SyncEngine()।
   - চ্যাট হ্যান্ডলারে: after save: sync_engine.sync_memories('to_render')।
   - from modules.error.ErrorLogger import ErrorLogger; error_logger = ErrorLogger(); if sync fails: error_logger.log_error(...) # টাস্ক ৮ সিঙ্ক।
   - from modules.approval.ApprovalManager import ApprovalManager; approval_manager = ApprovalManager(); if approval_manager.request_approval('sync', 'Sync memories'): ... # টাস্ক ৪ সিঙ্ক।
   - from modules.execution.ExecutionModule import ExecutionModule; executor = ExecutionModule(); executor.execute_cmd('curl ' + self.render_url) # টাস্ক ৫ সিঙ্ক, টেস্ট।

**ইন্টিগ্রেশন**: টাস্ক ৯-এর /status-এ sync স্টেট যোগ। টাস্ক ৮-এর ErrorLogger ইমপোর্ট করে সিঙ্ক এরর লগ। টাস্ক ৪-এর ApprovalManager ইমপোর্ট করে সিঙ্ক অ্যাপ্রুভ। টাস্ক ৫-এর execute_cmd ইউজ করে curl টেস্ট। টাস্ক ১-এর TaskMemoryManager ইউজ করে লোড/সেভ। পরবর্তী টাস্ক (e.g., ১১) এখানকার sync_memories কল করে। env 'SYNC_INTERVAL' ফিক্সড রাখো।

**টেস্টিং**:
1. tests/sync_test.py: class TestSyncEngine(unittest.TestCase): def test_sync(self): engine = SyncEngine(); memory_manager.save_memory('test', {'key': 'value'}, 'test'); result = engine.sync_memories(); assert result == "Success"; remote = requests.get(engine.render_url).json(); assert len(remote) > 0।
2. python -m unittest tests/sync_test.py।

**ফাইনাল আউটপুট**: SyncEngine.py কমপ্লিট, মেমরি সিঙ্ক কাজ করে। git commit -m "Task 10: Render Local Sync"।

---

#### **টাস্ক ১১: SelfUpgradeEngine Class Creation**

**ওভারভিউ**: SelfUpgradeEngine ক্লাস তৈরি করো যা সেলফ-আপগ্রেড ম্যানেজ করবে। অনুরোধ অ্যানালাইজ, কোড জেনারেট, ভ্যালিডেট, এক্সিকিউট। অ্যাপ্রুভাল, মেমরি, এক্সিকিউশন ইন্টিগ্রেট।

**প্রয়োজনীয় প্রিপারেশন**:
- টাস্ক ১০ কমপ্লিট (সিঙ্ক থেকে মেমরি শেয়ার)।
- লাইব্রেরী: ast (built-in), gitpython (pip install gitpython==3.1.30), requests (LLM), logging।
- env vars: os.environ['GIT_REPO'] = 'https://github.com/The-Mask-Of-Imran/The-Mask-Core-System'; os.environ['GROQ_API_KEY'] = 'your_key' # ফিক্সড।
- প্রম্পট টেমপ্লেট: "Generate Python code for feature {feature}: {description}" ফিক্সড।

**স্টেপ-বাই-স্টেপ ইমপ্লিমেন্টেশন**:
1. modules/upgrade/SelfUpgradeEngine.py ফাইল তৈরি।
2. class SelfUpgradeEngine:।
3. __init__ মেথড: self.git_repo = os.environ['GIT_REPO']; import git; self.repo = git.Repo('.') # লোকাল রেপো।
   - self.prompt_template = "Generate Python code for feature {feature}: {description}" # ফিক্সড।
4. upgrade_request মেথড: def upgrade_request(self, feature: str, description: str):।
   - from modules.approval.ApprovalManager import ApprovalManager; approval_manager = ApprovalManager()।
   - if not approval_manager.request_approval('upgrade', f"Upgrade for {feature}"): return "Denied"।
   - code = self._generate_code(feature, description)।
   - if not self._validate_code(code): return "Invalid code"।
   - self._execute_upgrade(code, feature)।
   - logging.info(f"Upgraded {feature}")।
5. _generate_code মেথড: def _generate_code(self, feature: str, description: str):।
   - prompt = self.prompt_template.format(feature=feature, description=description)।
   - from modules.router.ModelRouter import ModelRouter; router = ModelRouter(); return router.generate_response(prompt, 'code_gen') # Groq/Ollama দিয়ে।
6. _validate_code মেথড: def _validate_code(self, code: str):।
   - import ast; try: ast.parse(code); return True except SyntaxError as e: logging.error(f"Syntax error: {e}"); return False # ফিক্সড ভ্যালিডেশন।
7. _execute_upgrade মেথড: def _execute_upgrade(self, code: str, feature: str):।
   - with open(f'modules/{feature.lower()}/{feature}Module.py', 'w') as f: f.write(code) # ফিক্সড পাথ।
   - from modules.execution.ExecutionModule import ExecutionModule; executor = ExecutionModule(); executor.execute_cmd('pytest tests/') # লোকাল টেস্ট।
   - self.repo.git.add(A=True); self.repo.commit(m=f"Upgrade {feature}"); self.repo.git.push() # git ops।
   - from modules.sync.SyncEngine import SyncEngine; SyncEngine().sync_memories() # টাস্ক ১০ সিঙ্ক।
   - from modules.memory.TaskMemoryManager import TaskMemoryManager; memory_manager = TaskMemoryManager(); memory_manager.save_memory('upgrade', {'feature': feature, 'code': code}, 'upgrade') # টাস্ক ১ সিঙ্ক।
8. app.py-এ ইন্টিগ্রেট: from modules.upgrade.SelfUpgradeEngine import SelfUpgradeEngine।
   - upgrade_engine = SelfUpgradeEngine()।
   - চ্যাট হ্যান্ডলারে: if 'upgrade' in command: upgrade_engine.upgrade_request(feature, desc)।
   - from modules.error.ErrorLogger import ErrorLogger; error_logger = ErrorLogger(); if validate fails: error_logger.log_error(...) # টাস্ক ৮ সিঙ্ক।

**ইন্টিগ্রেশন**: টাস্ক ১০-এর SyncEngine ইমপোর্ট করে সিঙ্ক। টাস্ক ৮-এর ErrorLogger ইমপোর্ট করে এরর লগ। টাস্ক ৪-এর ApprovalManager ইমপোর্ট করে অ্যাপ্রুভ। টাস্ক ৫-এর ExecutionModule ইউজ করে টেস্ট রান। টাস্ক ১-এর save_memory কল। টাস্ক ২-৩-এর ModelRouter ইউজ করে কোড জেন। পরবর্তী টাস্ক (e.g., ১২) এখানকার execute_upgrade extend করে। env 'GIT_REPO' ফিক্সড রাখো।

**টেস্টিং**:
1. tests/upgrade_test.py: class TestSelfUpgradeEngine(unittest.TestCase): def test_request(self): engine = SelfUpgradeEngine(); code = engine._generate_code('dummy', 'test'); assert engine._validate_code(code)।
2. python -m unittest tests/upgrade_test.py।

**ফাইনাল আউটপুট**: SelfUpgradeEngine.py কমপ্লিট, অ্যাপগ্রেড রিকোয়েস্ট কাজ করে। git commit -m "Task 11: SelfUpgradeEngine"।

---

#### **টাস্ক ১২: Code Generate → Local Test → Git Push → Render Restart Flow**

**ওভারভিউ**: SelfUpgradeEngine-এ execute_flow মেথড যোগ করো যা কোড জেন, লোকাল টেস্ট, git পুশ, Render রিস্টার্ট করবে। ফেল হলে অ্যাবর্ট।

**প্রয়োজনীয় প্রিপারেশন**:
- টাস্ক ১১ কমপ্লিট (SelfUpgradeEngine বেস)।
- লাইব্রেরী: gitpython (pip install gitpython==3.1.30), subprocess (built-in), requests (Render), pytest (testing)।
- env vars: from config import RENDER_WEBHOOK   # config.py-তে যোগ করো: RENDER_WEBHOOK = os.getenv("RENDER_WEBHOOK", "your_webhook_url")
requests.post(RENDER_WEBHOOK)
 = 'your_webhook_url'; os.environ['GIT_CREDENTIALS'] = 'user:token' # ফিক্সড।

**স্টেপ-বাই-স্টেপ ইমপ্লিমেন্টেশন**:
1. modules/upgrade/SelfUpgradeEngine.py আপডেট (টাস্ক ১১-এর ফাইল)।
2. execute_flow মেথড যোগ: def execute_flow(self, feature: str, description: str):।
   - code = self._generate_code(feature, description)।
   - if not self._test_locally(code): return "Test failed" # ফেল অ্যাবর্ট।
   - self._git_push(feature)।
   - self._render_restart()।
   - logging.info("Flow completed")।
3. _test_locally মেথড: def _test_locally(self, code: str):।
   - test_file = f'tests/{feature.lower()}_test.py'; with open(test_file, 'w') as f: f.write('# Dummy test\nassert True') # ফিক্সড ডামি টেস্ট।
   - import subprocess; result = subprocess.run(['pytest', test_file], capture_output=True); return result.returncode == 0 # ইন্টিগ্রেশন/ইউনিট লজিক।
4. _git_push মেথড: def _git_push(self, feature: str):।
   - import git; repo = git.Repo('.')।
   - repo.git.add(A=True); repo.commit(m=f"Upgrade flow for {feature}")।
   - repo.git.push() # ক্রেডেনশিয়াল env দিয়ে।
5. _render_restart মেথড: def _render_restart(self): requests.post(os.environ['RENDER_WEBHOOK']) # ফিক্সড API কল।
6. app.py-এ ইন্টিগ্রেট: upgrade_engine = SelfUpgradeEngine()।
   - চ্যাট হ্যান্ডলারে: upgrade_engine.execute_flow(feature, desc)।
   - from modules.sync.SyncEngine import SyncEngine; SyncEngine().sync_memories() after push # টাস্ক ১০ সিঙ্ক।
   - from modules.error.ErrorLogger import ErrorLogger; error_logger = ErrorLogger(); if test fails: error_logger.log_error(...) # টাস্ক ৮ সিঙ্ক।
   - from modules.approval.ApprovalManager import ApprovalManager; approval_manager = ApprovalManager(); if approval_manager.request_approval('flow', 'Execute upgrade flow'): ... # টাস্ক ৪ সিঙ্ক।

**ইন্টিগ্রেশন**: টাস্ক ১১-এর SelfUpgradeEngine extend। টাস্ক ১০-এর SyncEngine ইমপোর্ট করে পুশ পর সিঙ্ক। টাস্ক ৮-এর ErrorLogger ইমপোর্ট করে ফেল লগ। টাস্ক ৪-এর ApprovalManager ইমপোর্ট করে ফ্লো অ্যাপ্রুভ। পরবর্তী টাস্ক (e.g., ১৩) এখানকার execute_flow কল করে। env 'RENDER_WEBHOOK' ফিক্সড রাখো।

**টেস্টিং**:
1. tests/flow_test.py: class TestFlow(unittest.TestCase): def test_execute(self): engine = SelfUpgradeEngine(); result = engine.execute_flow('dummy', 'test desc'); assert result != "Test failed"।
2. python -m unittest tests/flow_test.py।

**ফাইনাল আউটপুট**: SelfUpgradeEngine.py আপডেট, আপগ্রেড ফ্লো কাজ করে। git commit -m "Task 12: Upgrade Flow"।


#### **টাস্ক ১৩: Backup System (Zip + Google Drive)**

**ওভারভিউ**: এই টাস্কে BackupManager ক্লাস তৈরি করো যা আপগ্রেডের আগে/পরে সিস্টেম ব্যাকআপ নেবে (কোড, DB, কনফিগ zip করে Google Drive-এ আপলোড)। অটো ট্রিগার, টাইমস্ট্যাম্পড ভার্সনিং, রিটেনশন পলিসি (লাস্ট ১০)।

**প্রয়োজনীয় প্রিপারেশন**:
- টাস্ক ১২ কমপ্লিট (SelfUpgradeEngine থেকে প্রি-আপগ্রেড ট্রিগার)।
- লাইব্রেরী: shutil (built-in), zipfile (built-in), google-api-python-client (pip install google-api-python-client==2.50.0), oauth2client (pip install oauth2client==4.1.3), logging।
- env vars:from config import GOOGLE_DRIVE_FOLDER_ID, SERVICE_ACCOUNT_JSON = 'your_folder_id'; os.environ['SERVICE_ACCOUNT_JSON'] = 'path/to/service_account.json' # ফিক্সড ক্রেডেনশিয়াল।
- Google Drive সার্ভিস অ্যাকাউন্ট সেটআপ: credentials.json ফাইল data/ ফোল্ডারে।

**স্টেপ-বাই-স্টেপ ইমপ্লিমেন্টেশন**:
1. modules/backup/BackupManager.py ফাইল তৈরি।
2. class BackupManager:।
3. __init__ মেথড: from googleapiclient.discovery import build; from google.oauth2 import service_account; credentials = service_account.Credentials.from_service_account_file(os.environ['SERVICE_ACCOUNT_JSON'], scopes=['https://www.googleapis.com/auth/drive']); self.drive_service = build('drive', 'v3', credentials=credentials); self.folder_id = os.environ['GOOGLE_DRIVE_FOLDER_ID']; self.retention = 10 # ফিক্সড পলিসি।
4. create_backup মেথড: def create_backup(self):।
   - from datetime import datetime; timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')।
   - zip_name = f'backup_{timestamp}.zip'; import shutil; shutil.make_archive(zip_name[:-4], 'zip', '.') # রুট থেকে zip ফিক্সড।
   - file_metadata = {'name': zip_name, 'parents': [self.folder_id]}; media = googleapiclient.http.MediaFileUpload(zip_name, mimetype='application/zip')।
   - file = self.drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute(); logging.info(f"Uploaded backup {file['id']}") # আপলোড লজিক।
   - self._apply_retention()।
   - os.remove(zip_name) # ক্লিনআপ।
5. _apply_retention মেথড: def _apply_retention(self):।
   - query = f"'{self.folder_id}' in parents and mimeType='application/zip'"; files = self.drive_service.files().list(q=query, orderBy='createdTime desc').execute().get('files', [])।
   - if len(files) > self.retention: for file in files[self.retention:]: self.drive_service.files().delete(fileId=file['id']).execute() # লাস্ট ১০ রাখো ফিক্সড।
6. SelfUpgradeEngine.py আপডেট (টাস্ক ১১-১২): execute_flow-এ শুরুতে/শেষে: from modules.backup.BackupManager import BackupManager; BackupManager().create_backup() # প্রি/পোস্ট ট্রিগার ফিক্সড।
7. app.py-এ ইন্টিগ্রেট: from modules.backup.BackupManager import BackupManager।
   - backup_manager = BackupManager()।
   - চ্যাট হ্যান্ডলারে: if 'backup' in command: backup_manager.create_backup()।
   - from modules.sync.SyncEngine import SyncEngine; SyncEngine().sync_memories() after backup # টাস্ক ১০ সিঙ্ক।
   - from modules.error.ErrorLogger import ErrorLogger; error_logger = ErrorLogger(); if upload fails: error_logger.log_error(...) # টাস্ক ৮ সিঙ্ক।
   - from modules.approval.ApprovalManager import ApprovalManager; approval_manager = ApprovalManager(); if approval_manager.request_approval('backup', 'Create backup'): ... # টাস্ক ৪ সিঙ্ক।
   - from modules.memory.TaskMemoryManager import TaskMemoryManager; memory_manager = TaskMemoryManager(); memory_manager.save_memory('backup', {'timestamp': timestamp}, 'backup') # টাস্ক ১ সিঙ্ক।

**ইন্টিগ্রেশন**: টাস্ক ১২-এর SelfUpgradeEngine-এ create_backup কল। টাস্ক ১০-এর SyncEngine ইমপোর্ট করে ব্যাকআপ পর সিঙ্ক। টাস্ক ৮-এর ErrorLogger ইমপোর্ট করে আপলোড এরর লগ। টাস্ক ৪-এর ApprovalManager ইমপোর্ট করে ব্যাকআপ অ্যাপ্রুভ। টাস্ক ১-এর save_memory কল। পরবর্তী টাস্ক (e.g., ১৪) এখানকার create_backup ইউজ করে। env 'GOOGLE_DRIVE_FOLDER_ID' ফিক্সড রাখো।

**টেস্টিং**:
1. tests/backup_test.py: class TestBackupManager(unittest.TestCase): def test_create(self): manager = BackupManager(); manager.create_backup(); files = manager.drive_service.files().list(...).execute(); assert len(files['files']) == 1।
2. python -m unittest tests/backup_test.py।

**ফাইনাল আউটপুট**: BackupManager.py কমপ্লিট, ব্যাকআপ আপলোড কাজ করে। git commit -m "Task 13: Backup System"।

---

#### **টাস্ক ১৪: Rollback Mechanism (Git Checkout + Restore)**

ওভারভিউ: এই টাস্কে RollbackManager ক্লাস তৈরি করো যা আপগ্রেড ফেল হলে অটো/ম্যানুয়াল রোলব্যাক করবে। git checkout previous commit + BackupManager থেকে লেটেস্ট zip রিস্টোর + রিস্টার্ট। টাইমস্ট্যাম্প-বেসড ফাইল ম্যাচিং।
প্রয়োজনীয় প্রিপারেশন:
টাস্ক ১৩ কমপ্লিট (BackupManager থেকে timestamped zip)।
লাইব্রেরী: gitpython (pip install gitpython==3.1.30), shutil (built-in), subprocess (built-in), logging, datetime।
env vars: os.environ['RESTART_CMD'] = 'python app.py'; os.environ['BACKUP_DIR'] = 'backups/' # ফিক্সড।
backups/ ফোল্ডারে zip ফাইল সেভ হবে।
স্টেপ-বাই-স্টেপ ইমপ্লিমেন্টেশন:
modules/rollback/RollbackManager.py ফাইল তৈরি।
class RollbackManager:।
init মেথড: import git; self.repo = git.Repo('.'); self.backup_dir = os.environ['BACKUP_DIR']।
rollback মেথড: def rollback(self, reason: str = "Upgrade failed"):।
logging.warning(f"Rollback triggered: {reason}")।
prev_commit = self._get_previous_commit()。
if prev_commit: self.repo.git.checkout(prev_commit); logging.info(f"Reverted to commit {prev_commit}")।
self._restore_latest_backup()।
self._restart_system()।
from modules.sync.SyncEngine import SyncEngine; SyncEngine().sync_memories() # টাস্ক ১০ সিঙ্ক।
return "Rollback completed"।
_get_previous_commit মেথড: def _get_previous_commit(self):।
if self.repo.head.commit.parents: return self.repo.head.commit.parents[0].hexsha。
return None # ফিক্সড।
_restore_latest_backup মেথড: def _restore_latest_backup(self):।
from datetime import datetime; import glob, os。
backup_files = glob.glob(os.path.join(self.backup_dir, 'backup_*.zip'))。
if not backup_files: logging.error("No backup found"); return。
latest_file = max(backup_files, key=os.path.getctime) # লেটেস্ট টাইমস্ট্যাম্প ফিক্সড।
import shutil; shutil.unpack_archive(latest_file, '.')。
logging.info(f"Restored from {latest_file}")।
_restart_system মেথড: def _restart_system(self): subprocess.Popen(os.environ['RESTART_CMD'].split()) # ফিক্সড।
ErrorLogger.py আপডেট: log_error-এ: if classification == 'fatal' and 'upgrade' in context: RollbackManager().rollback(context)。
app.py-এ ইন্টিগ্রেট: from modules.rollback.RollbackManager import RollbackManager।
rollback_manager = RollbackManager()।
চ্যাটে 'rollback' কমান্ডে: rollback_manager.rollback("Manual request")।
from modules.memory.TaskMemoryManager import TaskMemoryManager; memory_manager = TaskMemoryManager(); memory_manager.save_memory('rollback', {'reason': reason, 'commit': prev_commit}, 'recovery') # টাস্ক ১ সিঙ্ক।
ইন্টিগ্রেশন: টাস্ক ১৩-এর BackupManager-এর timestamped zip ফরম্যাট ম্যাচ করে রিস্টোর। টাস্ক ১০ সিঙ্ক। টাস্ক ৮ এরর থেকে ট্রিগার। টাস্ক ১ মেমরি সেভ।
টেস্টিং:
tests/rollback_test.py: manager = RollbackManager(); manager.rollback("Test"); assert "Restored" in log_output।
python -m pytest tests/rollback_test.py।
ফাইনাল আউটপুট: RollbackManager.py কমপ্লিট, টাইমস্ট্যাম্প ম্যাচিংসহ রোলব্যাক কাজ করে। git commit -m "Task 14: Rollback Mechanism (Updated)"।


---

#### **টাস্ক ১৫: Auto Upgrade Proposal + Permission UI**

**ওভারভিউ**: এই টাস্কে ProposalGenerator ক্লাস তৈরি করো যা অটো আপগ্রেড প্রস্তাব করবে (এরর থেকে) এবং UI (কনসোল/Streamlit) দিয়ে পারমিশন নেবে। প্রস্তাবে ডিটেইলস (what, why, risk)।

**প্রয়োজনীয় প্রিপারেশন**:
- টাস্ক ১৪ কমপ্লিট (Rollback থেকে ফেল হ্যান্ডেল)।
- লাইব্রেরী: streamlit (pip install streamlit==1.10.0), json (built-in), logging।
- env vars: from config import LEARNING_PROMPT, STATE_SAVE_INTERVAL, ... = '0.5' # স্কোর ফিক্সড।
- Streamlit অ্যাপ: streamlit run ui/proposal_ui.py # ফিক্সড।

**স্টেপ-বাই-স্টেপ ইমপ্লিমেন্টেশন**:
1. modules/proposal/ProposalGenerator.py ফাইল তৈরি।
2. class ProposalGenerator:।
3. __init__ মেথড: self.threshold = float(os.environ['PROPOSAL_THRESHOLD'])।
4. generate_proposal মেথড: def generate_proposal(self):।
   - from modules.memory.TaskMemoryManager import TaskMemoryManager; memory_manager = TaskMemoryManager(); errors = memory_manager.load_memory('learning') # এরর থেকে অ্যানালাইজ ফিক্সড।
   - if len(errors) > 0 and self._score_proposal(errors) > self.threshold: proposal = {'what': 'Fix errors', 'why': 'From learned lessons', 'risk': 'Low'}।
     - logging.info(f"Generated proposal: {proposal}")।
     - return proposal।
   - return None।
5. _score_proposal মেথড: def _score_proposal(self, errors: list): return len(errors) / 10.0 # সিম্পল স্কোর ফিক্সড।
6. request_permission মেথড: def request_permission(self, proposal: dict):।
   - from modules.approval.ApprovalManager import ApprovalManager; approval_manager = ApprovalManager()।
   - return approval_manager.request_approval('proposal', json.dumps(proposal)) # ইন্টিগ্রেট।
7. UI স্ক্রিপ্ট: ui/proposal_ui.py ফাইল তৈরি: import streamlit as st; st.title('Upgrade Proposal'); proposal = ProposalGenerator().generate_proposal(); if proposal: st.json(proposal); if st.button('Approve'): st.write('Approved') # ওয়েব UI ফিক্সড।
   - কনসোল ফলব্যাক: print(json.dumps(proposal)); input('Approve? Y/N')।
8. app.py-এ ইন্টিগ্রেট: from modules.proposal.ProposalGenerator import ProposalGenerator।
   - proposal_gen = ProposalGenerator()।
   - চ্যাট হ্যান্ডলারে: proposal = proposal_gen.generate_proposal(); if proposal and proposal_gen.request_permission(proposal): upgrade_engine.upgrade_request(...) # টাস্ক ১১-১২ সিঙ্ক।
   - from modules.rollback.RollbackManager import RollbackManager; if not approved: RollbackManager().rollback() # টাস্ক ১৪ সিঙ্ক।
   - from modules.error.ErrorLogger import ErrorLogger; error_logger = ErrorLogger(); error_logger.log_error(...) if proposal irrelevant # টাস্ক ৮ সিঙ্ক।
   - from modules.sync.SyncEngine import SyncEngine; SyncEngine().sync_memories() after proposal # টাস্ক ১০ সিঙ্ক।

**ইন্টিগ্রেশন**: টাস্ক ১৪-এর RollbackManager ইমপোর্ট করে না-অ্যাপ্রুভড রোলব্যাক। টাস্ক ১১-১২-এর SelfUpgradeEngine কল করে অ্যাপ্রুভ পর। টাস্ক ৮-এর ErrorLogger ইমপোর্ট করে প্রপোজাল এরর লগ। টাস্ক ১০-এর SyncEngine ইমপোর্ট করে প্রপোজাল পর সিঙ্ক। টাস্ক ৪-এর ApprovalManager ইউজ করে। টাস্ক ১-এর TaskMemoryManager থেকে errors লোড। পরবর্তী টাস্ক (e.g., ১৬) এখানকার generate_proposal ইউজ করে। env 'PROPOSAL_THRESHOLD' ফিক্সড রাখো।

**টেস্টিং**:
1. tests/proposal_test.py: class TestProposalGenerator(unittest.TestCase): def test_generate(self): gen = ProposalGenerator(); proposal = gen.generate_proposal(); assert proposal is not None if errors exist।
2. python -m unittest tests/proposal_test.py।

**ফাইনাল আউটপুট**: ProposalGenerator.py কমপ্লিট, প্রপোজাল এবং UI কাজ করে। git commit -m "Task 15: Auto Upgrade Proposal"।



#### **টাস্ক ১৬: First Self-Upgrade Test (e.g., Add New /tts Endpoint)**

**ওভারভিউ**: এই টাস্কে সেলফ-আপগ্রেড ইঞ্জিনের প্রথম এন্ড-টু-এন্ড টেস্ট স্ক্রিপ্ট তৈরি করো। টেস্ট: upgrade_request("Add TTS endpoint") কল করে /tts এন্ডপয়েন্ট যোগ হয় কিনা চেক করো। রিপোর্ট জেনারেট: সাকসেস/ফেল, টাইম, এরর। ডামি TTS কোড টেমপ্লেট ব্যবহার।

**প্রয়োজনীয় প্রিপারেশন**:
- টাস্ক ১৫ কমপ্লিট (ProposalGenerator থেকে প্রপোজাল ট্রিগার)।
- লাইব্রেরী: pytest (pip install pytest==7.2.0), requests (verification), logging, time (built-in)।
- env vars: os.environ['TEST_TTS_ENDPOINT'] = 'http://localhost:5000/tts' # ফিক্সড টেস্ট URL।
- GitHub রেপো ক্লোন: git clone https://github.com/The-Mask-Of-Imran/The-Mask-Core-System।
- ডিরেক্টরি স্ট্রাকচার: tests/upgrade/ ফোল্ডার তৈরি যদি না থাকে।

**স্টেপ-বাই-স্টেপ ইমপ্লিমেন্টেশন**:
1. tests/upgrade/first_self_upgrade_test.py ফাইল তৈরি।
2. import pytest, requests, time, logging; from modules.upgrade.SelfUpgradeEngine import SelfUpgradeEngine।
3. class TestSelfUpgrade(unittest.TestCase):।
4. def setUp(self): self.engine = SelfUpgradeEngine(); self.start_time = time.time() # ফিক্সড টাইমিং।
5. def test_add_tts_endpoint(self):।
   - feature = "TTS Endpoint"; description = "Add /tts endpoint using gtts library to convert text to speech"।
   - result = self.engine.upgrade_request(feature, description) # টাস্ক ১১ কল।
   - assert result == "Success" or "Upgraded" in str(result) # ফেল হলে অ্যাসার্ট ফেল।
   - time.sleep(2) # সার্ভার রিস্টার্ট ওয়েট ফিক্সড।
   - response = requests.get(os.environ['TEST_TTS_ENDPOINT'] + '?text=Test speech')।
   - assert response.status_code == 200 and 'audio' in response.headers.get('Content-Type', '') # ভেরিফিকেশন লজিক।
   - elapsed = time.time() - self.start_time; logging.info(f"Test completed in {elapsed}s")।
6. ডামি TTS কোড টেমপ্লেট (SelfUpgradeEngine-এ হার্ডকোড বা প্রম্পটে ফোর্স): 
   - প্রম্পটে অ্যাড: "Use gtts library, add @app.route('/tts') def tts(): from gtts import gTTS; text = request.args.get('text'); tts = gTTS(text); tts.save('temp.mp3'); return send_file('temp.mp3')" ফিক্সড।
7. রিপোর্ট জেনারেট: def generate_report(success, elapsed, error=None): report = {'success': success, 'time': elapsed, 'error': error}; with open('reports/tts_upgrade_test.json', 'w') as f: json.dump(report, f)।
   - টেস্ট শেষে কল।
8. app.py-এ ইন্টিগ্রেট: if __name__ == '__main__' and 'test_upgrade' in sys.argv: pytest.main(['tests/upgrade/first_self_upgrade_test.py']) # ম্যানুয়াল রান ফিক্সড।
   - from modules.proposal.ProposalGenerator import ProposalGenerator; proposal_gen = ProposalGenerator(); if proposal_gen.generate_proposal(): pytest.main(...) # টাস্ক ১৫ ট্রিগার।
   - from modules.memory.TaskMemoryManager import TaskMemoryManager; memory_manager = TaskMemoryManager(); memory_manager.save_memory('tts_test', report, 'testing') # টাস্ক ১ সিঙ্ক।
   - from modules.error.ErrorLogger import ErrorLogger; error_logger = ErrorLogger(); if fails: error_logger.log_error(...) # টাস্ক ৮ সিঙ্ক।

**ইন্টিগ্রেশন**: টাস্ক ১৫-এর ProposalGenerator থেকে ট্রিগার। টাস্ক ১১-১২-এর SelfUpgradeEngine কল। টাস্ক ৮-এর ErrorLogger ইমপোর্ট করে ফেল লগ। টাস্ক ১-এর save_memory কল। পরবর্তী টাস্ক (e.g., ১৭) এখানকার টেস্ট রেজাল্ট ইউজ করে। env 'TEST_TTS_ENDPOINT' ফিক্সড রাখো।

**টেস্টিং**:
1. python -m pytest tests/upgrade/first_self_upgrade_test.py -v # রান করে চেক।
2. reports/tts_upgrade_test.json চেক: success=true হওয়া উচিত।

**ফাইনাল আউটপুট**: first_self_upgrade_test.py কমপ্লিট, প্রথম আপগ্রেড টেস্ট সাকসেসফুল। git commit -m "Task 16: First Self-Upgrade Test"।

---

#### **টাস্ক ১৭: Error Handling Improvement (Auto Rollback on Upgrade Fail)**

**ওভারভিউ**: এই টাস্কে ErrorLogger-কে আপগ্রেড করো যাতে আপগ্রেড ফেলে অটো রোলব্যাক ট্রিগার হয়। এরর ক্লাসিফাই (recoverable vs fatal), ট্রানজিয়েন্টের জন্য রিট্রাই, নোটিফিকেশন।

**প্রয়োজনীয় প্রিপারেশন**:
- টাস্ক ১৬ কমপ্লিট (টেস্ট থেকে ফেল সিমুলেট)।
- লাইব্রেরী: tenacity (pip install tenacity==8.0.1), logging, smtplib (built-in, ইমেইল নোটিফিকেশন) বা requests (Telegram API)।
- env vars: from config import LEARNING_PROMPT, STATE_SAVE_INTERVAL, ... = 'your_token'; os.environ['NOTIFY_CHAT_ID'] = 'your_chat_id' # ফিক্সড টেলিগ্রাম।
- Retry config: max_attempts=3 ফিক্সড।

**স্টেপ-বাই-স্টেপ ইমপ্লিমেন্টেশন**:
1. modules/error/ErrorLogger.py আপডেট (টাস্ক ৮-এর ফাইল)।
2. classify_error মেথড যোগ: def classify_error(self, e: Exception):।
   - recoverable = ['ConnectionError', 'TimeoutError']; fatal = ['SyntaxError', 'ImportError'] # ফিক্সড ক্লাসিফিকেশন।
   - error_type = type(e).__name__; if error_type in recoverable: return 'recoverable'; elif error_type in fatal: return 'fatal'; else: return 'unknown'।
3. log_error মেথড আপডেট: def log_error(self, e: Exception, context: str):।
   - classification = self.classify_error(e)।
   - logging.error(f"{classification.upper()}: {str(e)} in {context}")।
   - from tenacity import retry, stop_after_attempt, wait_fixed; if classification == 'recoverable': @retry(stop=stop_after_attempt(3), wait=wait_fixed(2)) def retry_action(): ... # রিট্রাই লজিক ফিক্সড।
   - if classification == 'fatal': from modules.rollback.RollbackManager import RollbackManager; RollbackManager().rollback() # অটো রোলব্যাক।
   - self._notify(f"Fatal error in upgrade: {str(e)}") # নোটিফিকেশন।
4. _notify মেথড: def _notify(self, message: str): requests.post(f"https://api.telegram.org/bot{os.environ['NOTIFY_TELEGRAM_TOKEN']}/sendMessage", data={'chat_id': os.environ['NOTIFY_CHAT_ID'], 'text': message}) # টেলিগ্রাম ফিক্সড।
5. SelfUpgradeEngine.py আপডেট (টাস্ক ১১-১২): execute_flow-এ try-except: except Exception as e: error_logger.log_error(e, 'Upgrade flow') # ট্রিগার ফিক্সড।
6. app.py-এ ইন্টিগ্রেট: error_logger = ErrorLogger()।
   - গ্লোবাল try-except-এ কল।
   - from modules.proposal.ProposalGenerator import ProposalGenerator; proposal_gen = ProposalGenerator(); if proposal_gen.generate_proposal(): ... error_logger.log_error(...) # টাস্ক ১৫ সিঙ্ক।
   - from modules.memory.TaskMemoryManager import TaskMemoryManager; memory_manager = TaskMemoryManager(); memory_manager.save_memory('error', {'type': classification}, 'error') # টাস্ক ১ সিঙ্ক।

**ইন্টিগ্রেশন**: টাস্ক ১৬-এর টেস্টে ফেল সিমুলেট করে ট্রিগার। টাস্ক ১৪-এর RollbackManager কল। টাস্ক ১৫-এর ProposalGenerator-এ এরর লগ। টাস্ক ১-এর save_memory কল। পরবর্তী টাস্ক (e.g., ১৮) এখানকার ক্লাসিফিকেশন ইউজ করে। env 'NOTIFY_TELEGRAM_TOKEN' ফিক্সড রাখো।

**টেস্টিং**:
1. tests/error_improve_test.py: class TestErrorHandling(unittest.TestCase): def test_fatal_rollback(self): try: raise SyntaxError("Test fatal") except Exception as e: error_logger.log_error(e, 'Test'); assert "Rollback" in log_output # মক চেক।
2. python -m pytest tests/error_improve_test.py।

**ফাইনাল আউটপুট**: ErrorLogger.py আপডেট, অটো রোলব্যাক এবং নোটিফিকেশন কাজ করে। git commit -m "Task 17: Error Handling Improvement"।

---

#### **টাস্ক ১৮: Save Upgrade History in Long-Term Memory**

**ওভারভিউ**: এই টাস্কে আপগ্রেডের পর প্রত্যেক আপগ্রেডের ইতিহাস (upgrade_id, changes, status, timestamp) TaskMemoryManager-এ সেভ করো। কোয়েরি-এবল (by date/feature)।

**প্রয়োজনীয় প্রিপারেশন**:
- টাস্ক ১৭ কমপ্লিট (ErrorLogger থেকে ফেল হিস্ট্রি)।
- লাইব্রেরী: json (built-in), sqlite3 (via TaskMemoryManager), datetime (built-in)।
- DB schema আপডেট: TaskMemoryManager.py-এ upgrades টেবল যোগ (যদি না থাকে)।

**স্টেপ-বাই-স্টেপ ইমপ্লিমেন্টেশন**:
1. modules/memory/TaskMemoryManager.py আপডেট (টাস্ক ১-এর ফাইল)।
2. __init__-এ টেবল যোগ: self.cursor.execute('''CREATE TABLE IF NOT EXISTS upgrades (id INTEGER PRIMARY KEY, upgrade_id TEXT, changes TEXT, status TEXT, timestamp DATETIME)'''); self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_upgrade_timestamp ON upgrades (timestamp)') # ফিক্সড স্কিমা।
3. SelfUpgradeEngine.py আপডেট: execute_flow/upgrade_request শেষে:।
   - from datetime import datetime; timestamp = datetime.now()।
   - history = {'upgrade_id': feature.lower().replace(' ', '_'), 'changes': code_snippet, 'status': 'success' if success else 'failed', 'timestamp': timestamp.isoformat()}।
   - memory_manager = TaskMemoryManager(); memory_manager.cursor.execute("INSERT INTO upgrades (upgrade_id, changes, status, timestamp) VALUES (?, ?, ?, ?)", (history['upgrade_id'], json.dumps(history['changes']), history['status'], history['timestamp']))।
   - memory_manager.conn.commit() # ফিক্সড সেভ।
4. নতুন কোয়েরি মেথড যোগ: def get_upgrade_history(self, by_date: str = None, by_feature: str = None):।
   - query = "SELECT * FROM upgrades"; params = []।
   - if by_date: query += " WHERE timestamp LIKE ?"; params.append(f"{by_date}%")।
   - if by_feature: query += " WHERE upgrade_id = ?"; params.append(by_feature)।
   - self.cursor.execute(query, params); return [dict(row) for row in self.cursor.fetchall()] # কোয়েরি লজিক ফিক্সড।
5. app.py-এ ইন্টিগ্রেট: from modules.memory.TaskMemoryManager import TaskMemoryManager।
   - memory_manager = TaskMemoryManager()।
   - চ্যাট হ্যান্ডলারে: if 'history' in command: history = memory_manager.get_upgrade_history(); return str(history)।
   - from modules.proposal.ProposalGenerator import ProposalGenerator; proposal_gen = ProposalGenerator(); proposal_gen.generate_proposal() uses history = memory_manager.get_upgrade_history() # টাস্ক ১৫ সিঙ্ক।
   - from modules.error.ErrorLogger import ErrorLogger; error_logger = ErrorLogger(); error_logger.log_error(...) saves to upgrades if upgrade context # টাস্ক ১৭ সিঙ্ক।
   - from modules.sync.SyncEngine import SyncEngine; SyncEngine().sync_memories() after save # টাস্ক ১০ সিঙ্ক।

**ইন্টিগ্রেশন**: টাস্ক ১৭-এর ErrorLogger-এ আপগ্রেড কনটেক্সট সেভ। টাস্ক ১৫-এর ProposalGenerator-এ history লোড। টাস্ক ১০-এর SyncEngine ইমপোর্ট করে সিঙ্ক। টাস্ক ১-এর TaskMemoryManager extend। পরবর্তী টাস্ক (e.g., ১৯) এখানকার get_upgrade_history ইউজ করে।

**টেস্টিং**:
1. tests/history_test.py: class TestUpgradeHistory(unittest.TestCase): def test_save_retrieve(self): memory_manager.save_to_upgrades('test_id', {'code': 'pass'}, 'success'); history = memory_manager.get_upgrade_history(by_feature='test_id'); assert len(history) == 1।
2. python -m pytest tests/history_test.py।

**ফাইনাল আউটপুট**: TaskMemoryManager.py আপডেট, আপগ্রেড হিস্ট্রি সেভ এবং কোয়েরি কাজ করে। git commit -m "Task 18: Save Upgrade History"।

#### **টাস্ক ১৯: Smart Prompt Chaining (For Sensitive Tasks)**

**ওভারভিউ**: এই টাস্কে PromptChainer ক্লাস তৈরি করো যা সেনসিটিভ টাস্কের জন্য মাল্টি-স্টেপ প্রম্পট চেইনিং করবে (plan → review → execute)। এতে accuracy বাড়বে এবং এরর কমবে। সেনসিটিভ ফ্ল্যাগ চেক করে চেইনিং অ্যাকটিভেট।

**প্রয়োজনীয় প্রিপারেশন**:
- টাস্ক ১৮ কমপ্লিট (Upgrade History থেকে সেনসিটিভ কনটেক্সট)।
- লাইব্রেরী: requests (built-in LLM calls), logging। (langchain ইউজ না করে ম্যানুয়াল লুপ ফিক্সড, যাতে ডিপেন্ডেন্সি কম থাকে)।
- env vars: from config import LEARNING_PROMPT, STATE_SAVE_INTERVAL, ... = '3' # ফিক্সড স্টেপস (plan, review, execute)।
- GitHub রেপো ক্লোন: git clone https://github.com/The-Mask-Of-Imran/The-Mask-Core-System।
- ডিরেক্টরি স্ট্রাকচার: modules/prompt/ ফোল্ডার তৈরি যদি না থাকে।

**স্টেপ-বাই-স্টেপ ইমপ্লিমেন্টেশন**:
1. modules/prompt/PromptChainer.py ফাইল তৈরি।
2. class PromptChainer:।
3. __init__ মেথড: self.steps = int(os.environ['CHAIN_STEPS']); self.router = ModelRouter() # from modules.router.ModelRouter import ModelRouter।
4. chain_prompts মেথড: def chain_prompts(self, initial_prompt: str, is_sensitive: bool = False):।
   - if not is_sensitive: return self.router.generate_response(initial_prompt, 'chat') # সিম্পল ফলব্যাক।
   - outputs = []; current_prompt = initial_prompt।
   - for step in range(self.steps):।
     - if step == 0: prompt = f"Step 1 - Plan: {current_prompt}. Provide a detailed plan." # ফিক্সড স্টেপ ১।
     - elif step == 1: prompt = f"Step 2 - Review: Review this plan: {outputs[-1]}. Suggest improvements and check risks." # ফিক্সড স্টেপ ২।
     - else: prompt = f"Step 3 - Execute: Based on plan and review: {outputs[-1]}. Generate final code/action." # ফিক্সড স্টেপ ৩।
     - output = self.router.generate_response(prompt, 'sensitive_chain') # LLM কল।
     - outputs.append(output)。
     - if "error" in output.lower(): logging.warning(f"Chain broke at step {step+1}"); break # ব্রেক রিকভারি।
   - final_output = outputs[-1] if outputs else "Chain failed"।
   - logging.info(f"Chain completed with {len(outputs)} steps")।
   - return final_output।
5. app.py-এ ইন্টিগ্রেট: from modules.prompt.PromptChainer import PromptChainer।
   - chainer = PromptChainer()।
   - চ্যাট হ্যান্ডলারে: if 'sensitive' in task_type or approval_manager.request_approval('sensitive', 'Use chaining'): response = chainer.chain_prompts(user_input, is_sensitive=True)।
   - from modules.approval.ApprovalManager import ApprovalManager; approval_manager = ApprovalManager(); # টাস্ক ৪ সিঙ্ক।
   - from modules.memory.TaskMemoryManager import TaskMemoryManager; memory_manager = TaskMemoryManager(); memory_manager.save_memory('chain', {'input': initial_prompt, 'output': final_output}, 'chaining') # টাস্ক ১ সিঙ্ক।
   - from modules.error.ErrorLogger import ErrorLogger; error_logger = ErrorLogger(); if chain fails: error_logger.log_error(...) # টাস্ক ৮ সিঙ্ক।

**ইন্টিগ্রেশন**: টাস্ক ৪-এর ApprovalManager ইমপোর্ট করে সেনসিটিভ চেক। টাস্ক ১-এর TaskMemoryManager ইউজ করে চেইন হিস্ট্রি সেভ। টাস্ক ৮-এর ErrorLogger ইমপোর্ট করে ব্রেক লগ। টাস্ক ২-৩-এর ModelRouter ইউজ করে প্রম্পট কল। পরবর্তী টাস্ক (e.g., ২০) এখানকার chain_prompts ইউজ করে। env 'CHAIN_STEPS' ফিক্সড রাখো।

**টেস্টিং**:
1. tests/prompt_test.py: class TestPromptChainer(unittest.TestCase): def test_chain(self): chainer = PromptChainer(); result = chainer.chain_prompts("Sensitive: Delete file", is_sensitive=True); assert "final" in result.lower() # মক রেসপন্স চেক।
2. python -m pytest tests/prompt_test.py।

**ফাইনাল আউটপুট**: PromptChainer.py কমপ্লিট, সেনসিটিভ টাস্কে চেইনিং কাজ করে। git commit -m "Task 19: Smart Prompt Chaining"।

---

#### **টাস্ক ২০: Final Test: Check if System Adds TTS on "Add TTS" Command**

**ওভারভিউ**: এই টাস্কে ফাইনাল ইন্টিগ্রেশন টেস্ট স্ক্রিপ্ট তৈরি করো যা ইউজার কমান্ড "TTS যোগ করো" সিমুলেট করে চেক করবে সিস্টেম নিজে /tts এন্ডপয়েন্ট যোগ করে কিনা। ফুল ফ্লো ভ্যালিডেশন, মেট্রিক্স রিপোর্ট।

**প্রয়োজনীয় প্রিপারেশন**:
- টাস্ক ১৯ কমপ্লিট (PromptChainer থেকে সেনসিটিভ চেইন)।
- লাইব্রেরী: pytest (pip install pytest==7.2.0), requests (endpoint test), gtts (pip install gtts==2.3.2, TTS লাইব্রেরী), logging, time।
- env vars:from config import LEARNING_PROMPT, STATE_SAVE_INTERVAL, ...= 'http://localhost:5000/tts' # ফিক্সড।
- Render-এ ডেপ্লয় করে রিয়েল টেস্ট।

**স্টেপ-বাই-স্টেপ ইমপ্লিমেন্টেশন**:
1. tests/integration/final_tts_test.py ফাইল তৈরি।
2. import pytest, requests, time, logging; from modules.upgrade.SelfUpgradeEngine import SelfUpgradeEngine।
3. class TestFinalTTS(unittest.TestCase):।
4. def setUp(self): self.engine = SelfUpgradeEngine(); self.start_time = time.time()।
5. def test_add_tts_command(self):।
   - command = "TTS যোগ করো" # ফিক্সড কমান্ড সিমুলেশন।
   - proposal_gen = ProposalGenerator(); proposal = proposal_gen.generate_proposal() # টাস্ক ১৫।
   - if proposal and proposal_gen.request_permission(proposal): result = self.engine.execute_flow("TTS Endpoint", "Add TTS using gtts") # টাস্ক ১২ ফ্লো।
   - assert "success" in result.lower() or "added" in result.lower()।
   - time.sleep(5) # রিস্টার্ট ওয়েট ফিক্সড।
   - response = requests.get(os.environ['TTS_TEST_URL'] + '?text=Hello from TTS test')।
   - assert response.status_code == 200 and 'audio/mpeg' in response.headers.get('Content-Type', '') # ভেরিফিকেশন।
   - elapsed = time.time() - self.start_time; report = {'success': True, 'time': elapsed, 'metrics': {'endpoint_added': True}}; logging.info(f"Final TTS test: {report}")।
   - with open('reports/final_tts_test.json', 'w') as f: json.dump(report, f)।
6. টেস্ট রান: if __name__ == '__main__': pytest.main(['-v', __file__])।
7. app.py-এ ইন্টিগ্রেট: চ্যাট হ্যান্ডলারে command == "TTS যোগ করো" হলে ফুল ফ্লো ট্রিগার (প্রপোজাল → চেইন → আপগ্রেড → টেস্ট)।
   - from modules.memory.TaskMemoryManager import TaskMemoryManager; memory_manager = TaskMemoryManager(); memory_manager.save_memory('tts_final_test', report, 'testing') # টাস্ক ১ সিঙ্ক।
   - from modules.error.ErrorLogger import ErrorLogger; error_logger = ErrorLogger(); if fails: error_logger.log_error(...) # টাস্ক ১৭ সিঙ্ক।

**ইন্টিগ্রেশন**: টাস্ক ১৫-এর ProposalGenerator ট্রিগার। টাস্ক ১৯-এর PromptChainer সেনসিটিভ চেইন। টাস্ক ১২-এর execute_flow কল। টাস্ক ১৭-এর ErrorLogger ফেল হ্যান্ডেল। টাস্ক ১-এর save_memory কল। পরবর্তী টাস্ক (e.g., ২১) এখানকার টেস্ট রেজাল্ট ইউজ করে। env 'TTS_TEST_URL' ফিক্সড রাখো।

**টেস্টিং**:
1. python -m pytest tests/integration/final_tts_test.py -v # রান করে চেক।
2. reports/final_tts_test.json চেক: success=true, endpoint_added=true হওয়া উচিত।

**ফাইনাল আউটপুট**: final_tts_test.py কমপ্লিট, "TTS যোগ করো" কমান্ডে সিস্টেম অটো যোগ করে। git commit -m "Task 20: Final TTS Test"।

---

#### **টাস্ক ২১: SelfLearningManager (Save Lessons from Errors)**

**ওভারভিউ**: এই টাস্কে SelfLearningManager ক্লাস তৈরি করো যা এরর থেকে লেসন জেনারেট করে TaskMemoryManager-এ সেভ করবে। অটো ট্রিগার, ক্যাটাগরি ('code_error' ইত্যাদি), চেক_লেসনস মেথড দিয়ে অ্যাভয়েড।

**প্রয়োজনীয় প্রিপারেশন**:
- টাস্ক ২০ কমপ্লিট (Final Test থেকে এরর সিমুলেশন)।
- লাইব্রেরী: logging, json, requests (LLM analysis), datetime।
- env vars: os.environ['LEARNING_PROMPT'] = "Analyze this error: {error}. What lesson should the system learn? Categorize as code_error/performance_issue/security/etc." # ফিক্সড প্রম্পট।
- GitHub রেপো ক্লোন: git clone https://github.com/The-Mask-Of-Imran/The-Mask-Core-System।
- ডিরেক্টরি স্ট্রাকচার: modules/learning/ ফোল্ডার তৈরি যদি না থাকে।

**স্টেপ-বাই-স্টেপ ইমপ্লিমেন্টেশন**:
1. modules/learning/SelfLearningManager.py ফাইল তৈরি।
2. class SelfLearningManager:।
3. __init__ মেথড: self.prompt_template = os.environ['LEARNING_PROMPT']; self.router = ModelRouter() # from modules.router.ModelRouter import ModelRouter।
4. learn_from_error মেথড: def learn_from_error(self, error_msg: str, context: str):।
   - prompt = self.prompt_template.format(error=error_msg)।
   - lesson_response = self.router.generate_response(prompt, 'learning_analysis') # LLM কল ফিক্সড।
   - try: lesson_data = json.loads(lesson_response) # {"lesson": "...", "category": "code_error"} ফরম্যাট অ্যাসিউম।
   - except: lesson_data = {"lesson": lesson_response, "category": "unknown"}।
   - from modules.memory.TaskMemoryManager import TaskMemoryManager; memory_manager = TaskMemoryManager(); memory_manager.save_memory('lesson', lesson_data, lesson_data['category'])।
   - logging.info(f"Learned lesson: {lesson_data['lesson']}")।
5. check_lessons মেথড: def check_lessons(self, action: str):।
   - memory_manager = TaskMemoryManager(); lessons = memory_manager.load_memory('learning')।
   - for lesson in lessons: if action.lower() in lesson['content']['lesson'].lower(): return False, lesson['content']['lesson'] # অ্যাভয়েড।
   - return True, None।
6. ErrorLogger.py আপডেট (টাস্ক ৮): log_error-এ: from modules.learning.SelfLearningManager import SelfLearningManager; SelfLearningManager().learn_from_error(str(e), context) # ট্রিগার ফিক্সড।
7. app.py-এ ইন্টিগ্রেট: from modules.learning.SelfLearningManager import SelfLearningManager।
   - learning_manager = SelfLearningManager()।
   - before sensitive action: allowed, reason = learning_manager.check_lessons(action); if not allowed: return f"Avoided: {reason}"।
   - from modules.error.ErrorLogger import ErrorLogger; error_logger = ErrorLogger(); error_logger.log_error(...) calls learn_from_error # টাস্ক ৮ সিঙ্ক।
   - from modules.memory.TaskMemoryManager import TaskMemoryManager; memory_manager = TaskMemoryManager(); memory_manager.save_memory(...) # টাস্ক ১ সিঙ্ক।
   - from modules.proposal.ProposalGenerator import ProposalGenerator; proposal_gen = ProposalGenerator(); proposal uses lessons from check_lessons # টাস্ক ১৫ সিঙ্ক।

**ইন্টিগ্রেশন**: টাস্ক ৮-এর ErrorLogger-এ learn_from_error কল। টাস্ক ১-এর TaskMemoryManager ইউজ করে সেভ/লোড। টাস্ক ১৫-এর ProposalGenerator-এ চেক_লেসনস ইউজ। পরবর্তী টাস্ক (e.g., ২২) এখানকার check_lessons ইউজ করে। env 'LEARNING_PROMPT' ফিক্সড রাখো।

**টেস্টিং**:
1. tests/learning_test.py: class TestSelfLearningManager(unittest.TestCase): def test_learn_and_check(self): manager = SelfLearningManager(); manager.learn_from_error("Connection refused", "DB connect"); allowed, _ = manager.check_lessons("connect DB"); assert not allowed।
2. python -m pytest tests/learning_test.py।

**ফাইনাল আউটপুট**: SelfLearningManager.py কমপ্লিট, এরর থেকে লেসন সেভ এবং অ্যাভয়েড কাজ করে। git commit -m "Task 21: SelfLearningManager"।


#### **টাস্ক ২২: Long-Term Task State Management (Runs for Years)**

**ওভারভিউ**: এই টাস্কে TaskStateManager ক্লাস তৈরি করো যা লং-টার্ম টাস্কের স্টেট (প্রোগ্রেস, ভ্যারিয়েবলস, ডিপেন্ডেন্সি) ম্যানেজ করবে। পিরিয়ডিক চেকপয়েন্ট সেভ (প্রতি ঘণ্টা), রিস্টার্টে অটো রিজুম। বছরের পর বছর চলার জন্য ডিজাইন।

**প্রয়োজনীয় প্রিপারেশন**:
- টাস্ক ২১ কমপ্লিট (SelfLearningManager থেকে লার্নিং টাস্ক স্টেট)।
- লাইব্রেরী: sqlite3 (built-in), json (built-in), schedule (pip install schedule==1.1.0), threading (built-in), logging, gzip (built-in, কম্প্রেশন)।
- env vars:from config import LEARNING_PROMPT, STATE_SAVE_INTERVAL, ...= '3600' # সেকেন্ড (১ ঘণ্টা) ফিক্সড।
- GitHub রেপো ক্লোন: git clone https://github.com/The-Mask-Of-Imran/The-Mask-Core-System।
- ডিরেক্টরি স্ট্রাকচার: modules/taskstate/ ফোল্ডার তৈরি যদি না থাকে।

**স্টেপ-বাই-স্টেপ ইমপ্লিমেন্টেশন**:
1. modules/taskstate/TaskStateManager.py ফাইল তৈরি।
2. class TaskStateManager:।
3. __init__ মেথড: self.db_path = 'data/task_states.db'; self.conn = sqlite3.connect(self.db_path, check_same_thread=False); self.cursor = self.conn.cursor()।
   - self.cursor.execute('''CREATE TABLE IF NOT EXISTS task_states (task_id TEXT PRIMARY KEY, state TEXT, last_update DATETIME, dependencies TEXT, progress REAL)''')।
   - self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_task_last_update ON task_states (last_update)') # পারফরম্যান্স ফিক্সড।
   - self.conn.commit()।
   - import threading; threading.Thread(target=self._periodic_checkpoint, daemon=True).start()।
4. save_state মেথড: def save_state(self, task_id: str, state_dict: dict, dependencies: list = None, progress: float = 0.0):।
   - import json, gzip, datetime; compressed_state = gzip.compress(json.dumps(state_dict).encode('utf-8'))।
   - timestamp = datetime.datetime.now().isoformat()।
   - deps_str = json.dumps(dependencies or [])।
   - self.cursor.execute("INSERT OR REPLACE INTO task_states (task_id, state, last_update, dependencies, progress) VALUES (?, ?, ?, ?, ?)", 
     (task_id, compressed_state, timestamp, deps_str, progress))।
   - self.conn.commit(); logging.info(f"Saved state for task {task_id}")।
5. load_state মেথড: def load_state(self, task_id: str):।
   - self.cursor.execute("SELECT state, dependencies, progress FROM task_states WHERE task_id = ?", (task_id,))।
   - row = self.cursor.fetchone(); if not row: return None, [], 0.0।
   - import gzip, json; decompressed = gzip.decompress(row[0]).decode('utf-8'); state = json.loads(decompressed)।
   - deps = json.loads(row[1]); progress = row[2]。
   - return state, deps, progress # ভ্যালিডেশন: if deps not satisfied, log warning।
6. _periodic_checkpoint মেথড: def _periodic_checkpoint(self):।
   - import schedule, time; schedule.every(int(os.environ['STATE_SAVE_INTERVAL'])/60).minutes.do(self._checkpoint_all)।
   - while True: schedule.run_pending(); time.sleep(1)।
7. _checkpoint_all মেথড: def _checkpoint_all(self):।
   - self.cursor.execute("SELECT task_id FROM task_states WHERE progress < 1.0")।
   - for row in self.cursor.fetchall(): task_id = row[0]; # assume external dict of active tasks; if task_id in active_tasks: self.save_state(task_id, active_tasks[task_id]['state'], ...)।
   - logging.info("Periodic checkpoint completed")।
8. app.py-এ ইন্টিগ্রেট: from modules.taskstate.TaskStateManager import TaskStateManager।
   - state_manager = TaskStateManager()।
   - লং-টার্ম টাস্ক শুরুতে: state_manager.save_state('ongoing_learning', {'progress': 0.0, 'data': {}}, ['task1'])।
   - রিস্টার্টে: state, deps, progress = state_manager.load_state('ongoing_learning'); if state: resume_task(state)।
   - from modules.memory.TaskMemoryManager import TaskMemoryManager; memory_manager = TaskMemoryManager(); memory_manager.save_memory('task_state', {'task_id': task_id}, 'state') # টাস্ক ১ সিঙ্ক।
   - from modules.error.ErrorLogger import ErrorLogger; error_logger = ErrorLogger(); if load fails: error_logger.log_error(...) # টাস্ক ৮ সিঙ্ক।

**ইন্টিগ্রেশন**: টাস্ক ১-এর TaskMemoryManager ইউজ করে স্টেট হিস্ট্রি সেভ। টাস্ক ৮-এর ErrorLogger ইমপোর্ট করে লোড এরর লগ। পরবর্তী টাস্ক (e.g., ২৩) এখানকার load_state ইউজ করে হাইব্রিড লার্নিং রিজুম। env 'STATE_SAVE_INTERVAL' ফিক্সড রাখো।

**টেস্টিং**:
1. tests/taskstate_test.py: class TestTaskStateManager(unittest.TestCase): def test_save_load(self): manager = TaskStateManager(); manager.save_state('test_task', {'key': 'value'}, ['dep1'], 0.5); state, deps, prog = manager.load_state('test_task'); assert state['key'] == 'value' and prog == 0.5।
2. python -m pytest tests/taskstate_test.py।

**ফাইনাল আউটপুট**: TaskStateManager.py কমপ্লিট, লং-টার্ম স্টেট সেভ/লোড/রিজুম কাজ করে। git commit -m "Task 22: Long-Term Task State Management"।

---

#### **টাস্ক ২৩: Hybrid Learning (Learn from Local + Groq)**

**ওভারভিউ**: এই টাস্কে HybridLearner ক্লাস তৈরি করো যা লোকাল (Ollama 14B) এবং ক্লাউড (Groq 70B) থেকে লার্ন করে মার্জ করবে। লোকাল প্রায়োরিটি, অপটিমাইজড সুইচিং। SelfLearningManager-এ ইন্টিগ্রেট।

**প্রয়োজনীয় প্রিপারেশন**:
- টাস্ক ২২ কমপ্লিট (TaskStateManager থেকে লার্নিং স্টেট রিজুম)।
- লাইব্রেরী: ollama (pip install ollama==0.0.18), requests (Groq), json (merge), logging।
- env vars: from config import LEARNING_PROMPT, STATE_SAVE_INTERVAL, ...= 'your_key'; os.environ['HYBRID_CONFIDENCE_THRESHOLD'] = '0.7' # ফিক্সড।
- GitHub রেপো ক্লোন: git clone https://github.com/The-Mask-Of-Imran/The-Mask-Core-System।
- ডিরেক্টরি স্ট্রাকচার: modules/learning/ ফোল্ডারে HybridLearner.py।

**স্টেপ-বাই-স্টেপ ইমপ্লিমেন্টেশন**:
1. modules/learning/HybridLearner.py ফাইল তৈরি।
2. class HybridLearner:।
3. __init__ মেথড: self.local_conn = LocalModelConnector(); self.router = ModelRouter()।
4. learn_hybrid মেথড: def learn_hybrid(self, query: str):।
   - local_response = self.local_conn.generate_response(query) # লোকাল কল ফিক্সড।
   - cloud_prompt = f"Provide high-quality answer: {query}"; cloud_response = self.router.generate_response(cloud_prompt, 'learning', model='groq_70b') # Groq প্রায়োরিটি।
   - merged = self._merge_results(local_response, cloud_response)।
   - from modules.learning.SelfLearningManager import SelfLearningManager; SelfLearningManager().learn_from_error("No error", f"Learned: {merged}") # টাস্ক ২১ ইন্টিগ্রেট।
   - logging.info("Hybrid learning completed")。
   - return merged।
5. _merge_results মেথড: def _merge_results(self, local: str, cloud: str):।
   - # সিম্পল ভোট/কনফিডেন্স: if len(cloud) > len(local) * 1.5: return cloud else: return local + " (local enhanced)" ফিক্সড লজিক।
   - return f"Local: {local}\nCloud: {cloud}\nMerged: {cloud if 'confidence' in cloud.lower() else local}"।
6. app.py-এ ইন্টিগ্রেট: from modules.learning.HybridLearner import HybridLearner।
   - hybrid_learner = HybridLearner()।
   - চ্যাট/লার্নিং হ্যান্ডলারে: if 'learn' in command: response = hybrid_learner.learn_hybrid(command)।
   - from modules.taskstate.TaskStateManager import TaskStateManager; state_manager = TaskStateManager(); state_manager.save_state('hybrid_learning', {'last_query': query}, progress=1.0) # টাস্ক ২২ সিঙ্ক।
   - from modules.memory.TaskMemoryManager import TaskMemoryManager; memory_manager = TaskMemoryManager(); memory_manager.save_memory('hybrid', {'response': merged}, 'learning') # টাস্ক ১ সিঙ্ক।
   - from modules.error.ErrorLogger import ErrorLogger; error_logger = ErrorLogger(); if merge fails: error_logger.log_error(...) # টাস্ক ৮ সিঙ্ক।

**ইন্টিগ্রেশন**: টাস্ক ২২-এর TaskStateManager ইউজ করে লার্নিং স্টেট সেভ। টাস্ক ২১-এর SelfLearningManager-এ লেসন পাস। টাস্ক ১-এর TaskMemoryManager ইউজ করে সেভ। টাস্ক ৮-এর ErrorLogger ইমপোর্ট করে এরর লগ। পরবর্তী টাস্ক (e.g., ২৪) এখানকার learn_hybrid ইউজ করে সামারাইজেশন। env 'HYBRID_CONFIDENCE_THRESHOLD' ফিক্সড রাখো।

**টেস্টিং**:
1. tests/hybrid_test.py: class TestHybridLearner(unittest.TestCase): def test_learn(self): learner = HybridLearner(); result = learner.learn_hybrid("What is AI?"); assert len(result) > 10।
2. python -m pytest tests/hybrid_test.py।

**ফাইনাল আউটপুট**: HybridLearner.py কমপ্লিট, হাইব্রিড লার্নিং কাজ করে। git commit -m "Task 23: Hybrid Learning"।

---

#### **টাস্ক ২৪: Auto Summarization (Compact Old Memories)**

**ওভারভিউ**: এই টাস্কে Summarizer ক্লাস তৈরি করো যা পুরনো মেমরি (>৬ মাস) অটো সামারাইজ করে কমপ্যাক্ট করবে। অরিজিনাল আর্কাইভ, সামারি সেভ। পিরিয়ডিক (মাসিক)।

**প্রয়োজনীয় প্রিপারেশন**:
- টাস্ক ২৩ কমপ্লিট (HybridLearner থেকে লার্নিং ডাটা)।
- লাইব্রেরী: schedule (pip install schedule==1.1.0), json, sqlite3 (via TaskMemoryManager), logging।
- env vars: from config import LEARNING_PROMPT, STATE_SAVE_INTERVAL, ... = '6'; os.environ['SUMMARY_INTERVAL_DAYS'] = '30' # ফিক্সড।
- GitHub রেপো ক্লোন: git clone https://github.com/The-Mask-Of-Imran/The-Mask-Core-System।
- ডিরেক্টরি স্ট্রাকচার: modules/memory/ ফোল্ডারে Summarizer.py।

**স্টেপ-বাই-স্টেপ ইমপ্লিমেন্টেশন**:
1. modules/memory/Summarizer.py ফাইল তৈরি।
2. class Summarizer:।
3. __init__ মেতড: self.memory_manager = TaskMemoryManager(); self.age_months = int(os.environ['SUMMARY_AGE_MONTHS']); self.interval_days = int(os.environ['SUMMARY_INTERVAL_DAYS'])।
   - import threading; threading.Thread(target=self._periodic_summarize, daemon=True).start()।
4. summarize_memories মেথড: def summarize_memories(self):।
   - from datetime import datetime, timedelta; cutoff = (datetime.now() - timedelta(days=30*self.age_months)).isoformat()।
   - old_memories = self.memory_manager.cursor.execute("SELECT * FROM memories WHERE timestamp < ?", (cutoff,)).fetchall()।
   - if not old_memories: return "No old memories"।
   - prompt = "Summarize these old memories concisely, keep key points: " + json.dumps([dict(m) for m in old_memories])।
   - from modules.router.ModelRouter import ModelRouter; router = ModelRouter(); summary = router.generate_response(prompt, 'summarization')。
   - summary_entry = {'content': {'summary': summary, 'original_count': len(old_memories)}, 'category': 'summary'}।
   - self.memory_manager.save_memory('auto_summary', summary_entry, 'summary')।
   - self._archive_originals(old_memories)।
   - logging.info(f"Summarized {len(old_memories)} memories")।
5. _archive_originals মেথড: def _archive_originals(self, memories: list):।
   - for mem in memories: self.memory_manager.cursor.execute("UPDATE memories SET category = 'archived' WHERE id = ?", (mem[0],))।
   - self.memory_manager.conn.commit() # আর্কাইভ ফিক্সড।
6. _periodic_summarize মেথড: def _periodic_summarize(self):।
   - import schedule, time; schedule.every(self.interval_days).days.do(self.summarize_memories)।
   - while True: schedule.run_pending(); time.sleep(1)।
7. app.py-এ ইন্টিগ্রেট: from modules.memory.Summarizer import Summarizer।
   - summarizer = Summarizer()।
   - চ্যাট হ্যান্ডলারে: if 'summarize' in command: summarizer.summarize_memories()।
   - from modules.taskstate.TaskStateManager import TaskStateManager; state_manager = TaskStateManager(); state_manager.save_state('summarization', {'last_run': datetime.now().isoformat()}) # টাস্ক ২২ সিঙ্ক।
   - from modules.memory.TaskMemoryManager import TaskMemoryManager; memory_manager = TaskMemoryManager(); memory_manager.save_memory(...) # টাস্ক ১ সিঙ্ক।
   - from modules.error.ErrorLogger import ErrorLogger; error_logger = ErrorLogger(); if summarize fails: error_logger.log_error(...) # টাস্ক ৮ সিঙ্ক।

**ইন্টিগ্রেশন**: টাস্ক ২২-এর TaskStateManager ইউজ করে সামারাইজেশন স্টেট সেভ। টাস্ক ১-এর TaskMemoryManager extend করে আর্কাইভ। টাস্ক ৮-এর ErrorLogger ইমপোর্ট করে ফেল লগ। পরবর্তী টাস্ক (e.g., ২৫) এখানকার summarize_memories ইউজ করে। env 'SUMMARY_AGE_MONTHS' ফিক্সড রাখো।

**টেস্টিং**:
1. tests/summarizer_test.py: class TestSummarizer(unittest.TestCase): def test_summarize(self): summarizer = Summarizer(); summarizer.summarize_memories(); summaries = memory_manager.load_memory('summary'); assert len(summaries) > 0।
2. python -m pytest tests/summarizer_test.py।

**ফাইনাল আউটপুট**: Summarizer.py কমপ্লিট, অটো সামারাইজেশন কাজ করে। git commit -m "Task 24: Auto Summarization"।



#### **টাস্ক ২৫: Final Security Audit + Rule-Based Approval**

**ওভারভিউ**: এই টাস্কে SecurityAuditor ক্লাস তৈরি করো যা সম্পূর্ণ সিস্টেমের স্ট্যাটিক সিকিউরিটি অডিট করবে (bandit/pylint দিয়ে) এবং ApprovalManager-কে রুল-ইঞ্জিন দিয়ে আপগ্রেড করবে (predefined if-then rules, e.g., delete without backup denied)। অডিট রিপোর্ট জেনারেট (vulnerabilities সহ)।

**প্রয়োজনীয় প্রিপারেশন**:
- টাস্ক ২৪ কমপ্লিট (Summarizer থেকে মেমরি সিকিউরিটি)।
- লাইব্রেরী: bandit (pip install bandit==1.7.5), pylint (pip install pylint==2.15.0), pyyaml (pip install pyyaml==6.0), logging, reportlab (pip install reportlab==3.6.12, PDF রিপোর্ট)।
- env vars: from config import LEARNING_PROMPT, STATE_SAVE_INTERVAL, ... = 'config/security_rules.yaml' # ফিক্সড রুল ফাইল।
- rules.yaml তৈরি: actions: delete_file: requires_backup: true; upgrade_code: requires_approval: true # ফিক্সড রুলস।
- GitHub রেপো ক্লোন: git clone https://github.com/The-Mask-Of-Imran/The-Mask-Core-System।
- ডিরেক্টরি স্ট্রাকচার: modules/security/ ফোল্ডার তৈরি যদি না থাকে।

**স্টেপ-বাই-স্টেপ ইমপ্লিমেন্টেশন**:
1. modules/security/SecurityAuditor.py ফাইল তৈরি।
2. class SecurityAuditor:।
3. __init__ মেথড: self.rules_file = os.environ['RULES_FILE']।
4. run_audit মেথড: def run_audit(self):।
   - import bandit; from bandit.core import manager; b_mgr = manager.BanditManager(config=None, agg_type='file'); b_mgr.discover_files(['.'], recursive=True); b_mgr.run_tests(); issues = b_mgr.get_issue_list()।
   - pylint_output = subprocess.run(['pylint', '--output-format=text', '.'], capture_output=True, text=True).stdout।
   - report = {'bandit_issues': [str(i) for i in issues], 'pylint_output': pylint_output, 'timestamp': datetime.now().isoformat()}।
   - from reportlab.lib.pagesizes import letter; from reportlab.pdfgen import canvas; c = canvas.Canvas("reports/security_audit.pdf", pagesize=letter); c.drawString(100, 750, "Security Audit Report"); y = 700; for issue in report['bandit_issues']: c.drawString(100, y, issue); y -= 20; c.save() # PDF রিপোর্ট ফিক্সড।
   - logging.info("Audit completed"); return report।
5. ApprovalManager.py আপডেট (টাস্ক ৪-এর ফাইল):।
   - import yaml; with open(self.rules_file, 'r') as f: self.rules = yaml.safe_load(f)।
   - request_approval মেথডে: if action in self.rules: if self.rules[action].get('requires_backup', False) and not has_backup(): return False # রুল চেক ফিক্সড।
   - if self.rules[action].get('requires_approval', True): # প্রম্পট/অটো।
6. app.py-এ ইন্টিগ্রেট: from modules.security.SecurityAuditor import SecurityAuditor।
   - auditor = SecurityAuditor()।
   - চ্যাট হ্যান্ডলারে: if 'audit' in command: report = auditor.run_audit(); return str(report)।
   - from modules.approval.ApprovalManager import ApprovalManager; approval_manager = ApprovalManager(); approval_manager.request_approval(...) uses updated rules # টাস্ক ৪ সিঙ্ক।
   - from modules.memory.TaskMemoryManager import TaskMemoryManager; memory_manager = TaskMemoryManager(); memory_manager.save_memory('audit', report, 'security') # টাস্ক ১ সিঙ্ক।
   - from modules.error.ErrorLogger import ErrorLogger; error_logger = ErrorLogger(); if scan fails: error_logger.log_error(...) # টাস্ক ৮ সিঙ্ক।

**ইন্টিগ্রেশন**: টাস্ক ৪-এর ApprovalManager আপডেট করে রুল ইঞ্জিন যোগ। টাস্ক ১-এর TaskMemoryManager ইউজ করে অডিট রিপোর্ট সেভ। টাস্ক ৮-এর ErrorLogger ইমপোর্ট করে স্ক্যান এরর লগ। পরবর্তী টাস্ক (e.g., ২৬) এখানকার run_audit ইউজ করে প্লাগিন সিকিউরিটি চেক। env 'RULES_FILE' ফিক্সড রাখো।

**টেস্টিং**:
1. tests/security_test.py: class TestSecurityAuditor(unittest.TestCase): def test_audit(self): auditor = SecurityAuditor(); report = auditor.run_audit(); assert 'issues' in report。
   - def test_rule_check(self): approval_manager = ApprovalManager(); assert approval_manager.request_approval('delete_file', 'Test') == False if no backup。
2. python -m pytest tests/security_test.py।

**ফাইনাল আউটপুট**: SecurityAuditor.py কমপ্লিট, অডিট এবং রুল-বেসড অ্যাপ্রুভাল কাজ করে। git commit -m "Task 25: Final Security Audit"।

---

#### **টাস্ক ২৬: Plugin System Base Creation (For Future Use)**

**ওভারভিউ**: এই টাস্কে প্লাগিন সিস্টেমের বেস তৈরি করো। PluginManager ক্লাস দিয়ে ডাইনামিক লোড, রেজিস্ট্রি, BasePlugin abstract class। ফিউচারে TTS/hacking প্লাগিন যোগের জন্য।

**প্রয়োজনীয় প্রিপারেশন**:
- টাস্ক ২৫ কমপ্লিট (SecurityAuditor থেকে প্লাগিন সিকিউরিটি)।
- লাইব্রেরী: importlib (built-in), abc (built-in), logging।
- env vars: from config import LEARNING_PROMPT, STATE_SAVE_INTERVAL, ... = 'plugins/' # ফিক্সড ডিরেক্টরি।
- plugins/ ফোল্ডার তৈরি, dummy_plugin.py যোগ।
- GitHub রেপো ক্লোন: git clone https://github.com/The-Mask-Of-Imran/The-Mask-Core-System।

**স্টেপ-বাই-স্টেপ ইমপ্লিমেন্টেশন**:
1. modules/plugin/PluginManager.py ফাইল তৈরি।
2. from abc import ABC, abstractmethod; class BasePlugin(ABC): @abstractmethod def execute(self, *args, **kwargs): pass # ইন্টারফেস ফিক্সড।
3. class PluginManager:।
4. __init__ মেথড: self.plugins = {}; self.plugin_dir = os.environ['PLUGIN_DIR']; self._load_plugins()।
5. _load_plugins মেথড: def _load_plugins(self):।
   - import os, importlib.util; for file in os.listdir(self.plugin_dir): if file.endswith('.py') and file != '__init__.py': module_name = file[:-3]; spec = importlib.util.spec_from_file_location(module_name, os.path.join(self.plugin_dir, file)); module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); for attr in dir(module): obj = getattr(module, attr); if isinstance(obj, type) and issubclass(obj, BasePlugin) and obj != BasePlugin: plugin = obj(); self.plugins[plugin.name] = plugin # ডাইনামিক লোড ফিক্সড।
6. register_plugin মেথড: def register_plugin(self, name: str, plugin: BasePlugin): self.plugins[name] = plugin。
7. execute_plugin মেথড: def execute_plugin(self, name: str, *args, **kwargs): if name in self.plugins: return self.plugins[name].execute(*args, **kwargs); else: raise ValueError("Plugin not found")।
8. ডামি প্লাগিন: plugins/dummy_plugin.py: class DummyPlugin(BasePlugin): name = "dummy"; def execute(self, input): return f"Dummy executed with {input}"।
9. app.py-এ ইন্টিগ্রেট: from modules.plugin.PluginManager import PluginManager。
   - plugin_manager = PluginManager()।
   - চ্যাট হ্যান্ডলারে: if 'plugin' in command: result = plugin_manager.execute_plugin('dummy', 'test')।
   - from modules.security.SecurityAuditor import SecurityAuditor; auditor = SecurityAuditor(); auditor.run_audit() includes plugin dir # টাস্ক ২৫ সিঙ্ক।
   - from modules.memory.TaskMemoryManager import TaskMemoryManager; memory_manager = TaskMemoryManager(); memory_manager.save_memory('plugin_load', {'plugins': list(plugin_manager.plugins.keys())}, 'plugin') # টাস্ক ১ সিঙ্ক।
   - from modules.error.ErrorLogger import ErrorLogger; error_logger = ErrorLogger(); if load fails: error_logger.log_error(...) # টাস্ক ৮ সিঙ্ক।

**ইন্টিগ্রেশন**: টাস্ক ২৫-এর SecurityAuditor ইউজ করে প্লাগিন ডির স্ক্যান। টাস্ক ১-এর TaskMemoryManager ইউজ করে লোড হিস্ট্রি সেভ। টাস্ক ৮-এর ErrorLogger ইমপোর্ট করে লোড এরর লগ। পরবর্তী টাস্ক (e.g., ২৭) এখানকার execute_plugin ইউজ করে ভয়েস প্লাগিন। env 'PLUGIN_DIR' ফিক্সড রাখো।

**টেস্টিং**:
1. tests/plugin_test.py: class TestPluginManager(unittest.TestCase): def test_load_execute(self): manager = PluginManager(); result = manager.execute_plugin('dummy', 'test_input'); assert 'Dummy executed' in result।
2. python -m pytest tests/plugin_test.py।

**ফাইনাল আউটপুট**: PluginManager.py কমপ্লিট, প্লাগিন লোড/এক্সিকিউট কাজ করে। git commit -m "Task 26: Plugin System Base"।

---

#### **টাস্ক ২৭: Voice Interface Preparation (TTS/STT)**

**ওভারভিউ**: এই টাস্কে VoiceInterface ক্লাস তৈরি করো যা TTS (Text-to-Speech) এবং STT (Speech-to-Text) প্রোভাইড করবে। বেসিক wrappers, কোর চ্যাটে ভয়েস মোড ইন্টিগ্রেট।

**প্রয়োজনীয় প্রিপারেশন**:
- টাস্ক ২৬ কমপ্লিট (Plugin System থেকে ভয়েস প্লাগিন পটেনশিয়াল)।
- লাইব্রেরী: gtts (pip install gtts==2.3.2), pyttsx3 (pip install pyttsx3==2.90, অফলাইন TTS), speech_recognition (pip install SpeechRecognition==3.10.0), pydub (pip install pydub==0.25.1, অডিও হ্যান্ডলিং)।
- env vars: from config import LEARNING_PROMPT, STATE_SAVE_INTERVAL, ... = 'gtts'; os.environ['STT_ENGINE'] = 'speech_recognition' # ফিক্সড।
- GitHub রেপো ক্লোন: git clone https://github.com/The-Mask-Of-Imran/The-Mask-Core-System।
- ডিরেক্টরি স্ট্রাকচার: modules/voice/ ফোল্ডার তৈরি।

**স্টেপ-বাই-স্টেপ ইমপ্লিমেন্টেশন**:
1. modules/voice/VoiceInterface.py ফাইল তৈরি।
2. class VoiceInterface:।
3. __init__ মেথড: self.tts_engine = os.environ['TTS_ENGINE']; self.stt_engine = os.environ['STT_ENGINE']।
4. tts মেথড: def tts(self, text: str, output_file: str = 'output.mp3'):।
   - if self.tts_engine == 'gtts': from gtts import gTTS; tts = gTTS(text); tts.save(output_file)।
   - elif self.tts_engine == 'pyttsx3': import pyttsx3; engine = pyttsx3.init(); engine.save_to_file(text, output_file); engine.runAndWait()。
   - logging.info(f"TTS generated: {output_file}")。
   - return output_file।
5. stt মেথড: def stt(self, audio_file: str):।
   - import speech_recognition as sr; r = sr.Recognizer(); with sr.AudioFile(audio_file) as source: audio = r.record(source); try: text = r.recognize_google(audio); except: text = "Recognition failed"।
   - logging.info(f"STT result: {text}")।
   - return text।
6. integrate_with_chat মেথড: def integrate_with_chat(self, audio_input: str): text = self.stt(audio_input); response = chat_handler(text); audio_out = self.tts(response); return audio_out # কোর চ্যাট ইন্টিগ্রেশন ফিক্সড।
7. app.py-এ ইন্টিগ্রেট: from modules.voice.VoiceInterface import VoiceInterface।
   - voice_interface = VoiceInterface()।
   - @app.route('/voice', methods=['POST']): audio_file = request.files['audio']; text = voice_interface.stt(audio_file.filename); response_text = chat_handler(text); audio_out = voice_interface.tts(response_text); return send_file(audio_out)।
   - from modules.memory.TaskMemoryManager import TaskMemoryManager; memory_manager = TaskMemoryManager(); memory_manager.save_memory('voice', {'text': text, 'response': response_text}, 'voice') # টাস্ক ১ সিঙ্ক।
   - from modules.plugin.PluginManager import PluginManager; plugin_manager = PluginManager(); plugin_manager.execute_plugin('voice_plugin', text) if exists # টাস্ক ২৬ সিঙ্ক।
   - from modules.error.ErrorLogger import ErrorLogger; error_logger = ErrorLogger(); if recognition fails: error_logger.log_error(...) # টাস্ক ৮ সিঙ্ক।

**ইন্টিগ্রেশন**: টাস্ক ২৬-এর PluginManager ইউজ করে ভয়েস প্লাগিন এক্সিকিউট। টাস্ক ১-এর TaskMemoryManager ইউজ করে ভয়েস হিস্ট্রি সেভ। টাস্ক ৮-এর ErrorLogger ইমপোর্ট করে STT/TTS এরর লগ। পরবর্তী টাস্ক (e.g., ২৮) এখানকার integrate_with_chat ইউজ করে ড্যাশবোর্ডে ভয়েস। env 'TTS_ENGINE' ফিক্সড রাখো।

**টেস্টিং**:
1. tests/voice_test.py: class TestVoiceInterface(unittest.TestCase): def test_tts_stt(self): interface = VoiceInterface(); audio = interface.tts("Test text"); text = interface.stt(audio); assert "Test text" in text。
2. python -m pytest tests/voice_test.py।

**ফাইনাল আউটপুট**: VoiceInterface.py কমপ্লিট, TTS/STT প্রস্তুত এবং চ্যাটে ইন্টিগ্রেটেড। git commit -m "Task 27: Voice Interface Preparation"।


#### **টাস্ক ২৮: Dashboard UI (Streamlit)**

ওভারভিউ: Streamlit দিয়ে ইউজার-ফ্রেন্ডলি ড্যাশবোর্ড তৈরি করো। স্ট্যাটাস, মেমরি, আপগ্রেড হিস্ট্রি, অ্যাপ্রুভাল বাটন, pagination + অথেনটিকেশন।
প্রয়োজনীয় প্রিপারেশন:
টাস্ক ২৭ কমপ্লিট।
লাইব্রেরী: streamlit==1.29.0, pandas==2.0.3, requests, pyyaml।
env vars: DASHBOARD_AUTH_PASSWORD = 'secure_pass_2025' (ফিক্সড); STREAMLIT_PORT = '8501'।
dashboard/ ফোল্ডারে app.py।
স্টেপ-বাই-স্টেপ ইমপ্লিমেন্টেশন:
dashboard/app.py তৈরি।
import streamlit as st, pandas as pd, requests, yaml, os।
@st.cache_data def get_auth_password(): from config import DASHBOARD_AUTH_PASSWORD
if st.text_input("Password", type="password") != DASHBOARD_AUTH_PASSWORD:।
if st.text_input("Password", type="password") != get_auth_password(): st.error("Wrong password"); st.stop()।
st.set_page_config(page_title="The Mask Dashboard", layout="wide")।
st.sidebar.title("Navigation"); page = st.sidebar.radio("Go to", ["Status", "Memory Query", "Upgrade History", "Approvals"])।
if page == "Status": st.header("System Status"); resp = requests.get("http://localhost:5000/status", headers={"Auth-Key": os.environ['STATUS_AUTH_KEY']}); data = resp.json(); st.json(data); df = pd.DataFrame.from_dict(data['models'], orient='index'); st.dataframe(df)।
if page == "Memory Query": st.header("Long-Term Memory"); cat = st.text_input("Category filter"); if st.button("Load"): mem = requests.get("http://localhost:5000/memory", params={"category": cat}).json(); df = pd.DataFrame(mem); st.dataframe(df)।
if page == "Upgrade History": st.header("Upgrade History"); feat = st.text_input("Filter by feature"); if st.button("Show"): hist = requests.get("http://localhost:5000/upgrade_history", params={"feature": feat}).json(); df = pd.DataFrame(hist); st.dataframe(df)।
if page == "Approvals": st.header("Pending Approvals"); # API থেকে pending list আনা যেতে পারে; মক: items = [{"action": "Delete file X", "desc": "Requires backup"}]; for item in items: col1, col2 = st.columns(2); col1.write(item['action']); if col2.button("Approve"): st.success("Approved")।
app.py (মেইন)-এ: if 'open_dashboard' in command: subprocess.Popen(['streamlit', 'run', 'dashboard/app.py', '--server.port', os.environ['STREAMLIT_PORT']])।
from modules.memory.TaskMemoryManager import TaskMemoryManager; memory_manager = TaskMemoryManager(); memory_manager.save_memory('dashboard', {'page': page}, 'ui')。
ইন্টিগ্রেশন: টাস্ক ৯-এর /status-এ Auth-Key হেডার যোগ। টাস্ক ১৮ হিস্ট্রি কোয়েরি। টাস্ক ৪ অ্যাপ্রুভাল বাটন। টাস্ক ১ মেমরি লগ।
টেস্টিং:
streamlit run dashboard/app.py — পাসওয়ার্ড দিয়ে লগইন, সব পেজ চেক।
Auth-Key ছাড়া /status কল করলে 401 আসবে — ঠিক আছে।
ফাইনাল আউটপুট: dashboard/app.py কমপ্লিট, অথ + স্ট্যাটাস + হিস্ট্রি + অ্যাপ্রুভাল কাজ করে। git commit -m "Task 28: Dashboard UI (Updated with Auth)"।


টাস্ক ২৯: Full System End-to-End Test (1 Month Long Task)


ওভারভিউ: E2ETestSuite স্ক্রিপ্ট তৈরি করো। ৩০ দিনের লং টাস্ক সিমুলেশন (accelerated), স্টেবিলিটি, মেমরি, আপগ্রেড, সুইচিং টেস্ট। রিপোর্ট: task_29_e2e.json।
প্রয়োজনীয় প্রিপারেশন:
টাস্ক ২৮ কমপ্লিট।
লাইব্রেরী: pytest, requests, time, logging, json।
env vars: E2E_SIMULATION_SECONDS = '1800' # ৩০ মিনিট অ্যাক্সিলারেটেড (১ দিন = ৬০ সেকেন্ড)।
tests/e2e/e2e_suite.py।
from config import STATUS_AUTH_KEY
headers={'Auth-Key': STATUS_AUTH_KEY}


স্টেপ-বাই-স্টেপ ইমপ্লিমেন্টেশন:
tests/e2e/e2e_suite.py তৈরি।
import time, requests, json, logging, threading。
class E2ETestSuite:।
def init(self): self.report = {'start': time.time(), 'checks': [], 'errors': 0}; self.active = True।
def run_simulation(self):।
total_sec = int(os.environ['E2E_SIMULATION_SECONDS']); end = time.time() + total_sec。
while time.time() < end and self.active:।
try: status = requests.get('http://localhost:5000/status', headers={'Auth-Key': os.environ['STATUS_AUTH_KEY']}).json(); assert status['system_health'] == 'ok'。
mem_count = len(requests.get('http://localhost:5000/memory').json()); assert mem_count > 0।
upgrade = requests.post('http://localhost:5000/upgrade', json={'feature': 'e2e_test'}); assert 'success' in upgrade.text.lower()。
self.report['checks'].append({'time': time.time(), 'status': 'pass'})。
except Exception as e: self.report['errors'] += 1; self.report['checks'].append({'time': time.time(), 'status': 'fail', 'error': str(e)})।
time.sleep(60) # প্রতি মিনিট চেক।
self.report['duration'] = time.time() - self.report['start']; self.report['success_rate'] = (len(self.report['checks']) - self.report['errors']) / len(self.report['checks']) * 100 if self.report['checks'] else 0。
with open('reports/task_29_e2e.json', 'w') as f: json.dump(self.report, f, indent=2)。
app.py-এ: if 'run_e2e' in command: subprocess.Popen(['python', 'tests/e2e/e2e_suite.py'])।
from modules.memory.TaskMemoryManager import TaskMemoryManager; memory_manager.save_memory('e2e', self.report, 'testing')。
ইন্টিগ্রেশন: টাস্ক ৯ /status, টাস্ক ১৮ হিস্ট্রি, টাস্ক ১২ আপগ্রেড কল। টাস্ক ২৮ ড্যাশবোর্ডে রিপোর্ট দেখানো।
টেস্টিং:
python tests/e2e/e2e_suite.py — ৫ মিনিট চালিয়ে reports/task_29_e2e.json চেক।
success_rate > 95% হওয়া উচিত।
ফাইনাল আউটপুট: e2e_suite.py কমপ্লিট, রিপোর্ট task_29_e2e.json-এ সেভ। git commit -m "Task 29: Full E2E Test (Updated)"।


টাস্ক ৩০: v1.0 Release + Documentation
ওভারভিউ: v1.0 রিলিজ: ভার্সন বাম্প, changelog, sphinx ডকুমেন্টেশন, GitHub release, Render পুশ।
প্রয়োজনীয় প্রিপারেশন:
টাস্ক ২৯ পাস।
লাইব্রেরী: sphinx (pip install sphinx==5.3.0), gitpython, setuptools।
env vars: RELEASE_VERSION = '1.0.0'; GITHUB_TOKEN = 'ghp_xxx'।
docs/ ফোল্ডারে conf.py, index.rst।
from config import RELEASE_VERSION, GITHUB_TOKEN
স্টেপ-বাই-স্টেপ ইমপ্লিমেন্টেশন:
release.py তৈরি।
import git, os, subprocess; repo = git.Repo('.'); version = os.environ['RELEASE_VERSION']。
Version bump
with open('VERSION', 'w') as f: f.write(version)。
Changelog
log = repo.git.log('--pretty=format:%s', 'HEAD~10..HEAD'); with open('CHANGELOG.md', 'a') as f: f.write(f"\n## v{version}\n{log}")।
Docs
subprocess.run(['sphinx-apidoc', '-o', 'docs/source', '.'], check=True); subprocess.run(['make', 'html'], cwd='docs', check=True)।
GitHub release
repo.create_tag(f'v{version}'); repo.remotes.origin.push(f'v{version}')। subprocess.run(['gh', 'release', 'create', f'v{version}', '--title', f'v{version}', '--notes', 'CHANGELOG.md'], env={**os.environ, 'GITHUB_TOKEN': os.environ['GITHUB_TOKEN']})।
Render deploy
repo.git.push('render', 'main') # Render webhook auto-deploy।
app.py-এ: if 'release_v1' in sys.argv: subprocess.run(['python', 'release.py'])।
from modules.memory.TaskMemoryManager import TaskMemoryManager; memory_manager.save_memory('release_v1', {'version': version}, 'release')。
ইন্টিগ্রেশন: টাস্ক ২৯ টেস্ট পাস পর রিলিজ। টাস্ক ১ মেমরি সেভ।
টেস্টিং:
python release.py — VERSION, CHANGELOG, docs/_build/html, GitHub tag/release চেক।
Render-এ লাইভ সার্ভিস চেক।
ফাইনাল আউটপুট: v1.0 রিলিজড, ডকুমেন্টেশন + changelog + deploy কমপ্লিট। git commit -m "Task 30: v1.0 Release + Documentation (Final)"।









—---------------------------------------------------------------------------------------------------------------------



# টাস্ক ডকুমেন্টেশন: [টাস্ক নম্বর] - [টাস্ক নাম]

## 1. বেসিক ইনফো
- **টাস্ক নম্বর**: [e.g., 1]
- **টাস্ক নাম**: [e.g., Long-Term Memory Fix]
- **কমপ্লিশন তারিখ**: [e.g., YYYY-MM-DD]
- **ভার্সন**: [e.g., v0.1]
- **রিলেটেড ফেজ**: [e.g., ফেজ ১]
- **প্রিভিয়াস ডিপেন্ডেন্সি**: [e.g., কোনো টাস্ক নেই, অথবা টাস্ক ০ যদি থাকে]
- **পরবর্তী সম্ভাব্য টাস্ক**: [e.g., টাস্ক ২ - হাইব্রিড সুইচিং]

## 2. কাজের বিবরণ
- **কী কাজ করা হলো**: [বিস্তারিত বর্ণনা, e.g., লং-টার্ম মেমরি সিস্টেমের Connection refused এরর ফিক্স করা হয়েছে। এখন মেমরি স্টোর, রিট্রিভ এবং ব্যাকআপ কাজ করে।]
- **কেন করা হলো**: [উদ্দেশ্য, e.g., মেমরি স্থিতিশীল না হলে সেলফ-আপগ্রেড সম্ভব না, এবং লং-টার্ম টাস্ক মনে রাখা যাবে না।]
- **প্রভাব**: [সিস্টেমে কী চেঞ্জ, e.g., এখন সিস্টেম বছরের পর বছর ডাটা সেভ করতে পারে, এবং অন্য ফিচারস (e.g., সেলফ-লার্নিং) এর উপর ডিপেন্ড করে।]

## 3. ইমপ্লিমেন্টেশন ডিটেইলস
- **কীভাবে টাস্ক কমপ্লিট করা হলো**: [স্টেপ-বাই-স্টেপ, e.g.,
  1. এরর ডায়াগনোজ: পোর্ট কনফ্লিক্ট চেক।
  2. ক্লাস আপডেট: TaskMemoryManager-এ init মেথড যোগ।
  3. টেস্ট: স্যাম্পল ডাটা সেভ/লোড।
  4. ডেপ্লয়: Git push এবং Render রিস্টার্ট।]
- **সময়কাল**: [e.g., ২ দিন]
- **চ্যালেঞ্জস এবং সল্যুশন**: [e.g., Concurrency issue: WAL মোড চালু করা।]

## 4. ব্যবহৃত ফিচারস, লজিক এবং লাইব্রেরীস
- **ব্যবহৃত ফিচারস**: [লিস্ট, e.g., SQLite ডাটাবেস, JSON ব্যাকআপ, এরর রিট্রাই মেকানিজম।]
- **ব্যবহৃত লজিক**: [বিস্তারিত, e.g.,
  - রিট্রাই লজিক: Exponential backoff (tenacity লাইব্রেরী দিয়ে)।
  - ডাটা স্টোরেজ লজিক: JSON string-এ কনভার্ট করে SQLite-এ সেভ, ক্যাটাগরি-ভিত্তিক কোয়েরি।]
- **ব্যবহৃত লাইব্রেরীস/টুলস**: [লিস্ট সহ ভার্সন যদি থাকে, e.g.,
  - sqlite3 (built-in): ডাটাবেস কানেকশন।
  - json (built-in): ব্যাকআপ।
  - tenacity: রিট্রাই লুপ।
  - logging: এরর লগিং।]

## 5. কোড ডিটেইলস
- **প্রভাবিত ফাইলস**: [লিস্ট, e.g.,
  - TaskMemoryManager.py (প্রধান ফাইল)।
  - config.json (DB পাথ যোগ)।
  - tests/memory_test.py (টেস্ট স্ক্রিপ্ট)।]
- **কোড স্নিপেটস এবং ব্যাখ্যা**: [প্রত্যেক স্নিপেটের সাথে ব্যাখ্যা, যাতে নতুন AI বুঝতে পারে।
  ```python
  # TaskMemoryManager.py - init মেথড (লজিক: কানেকশন তৈরি এবং এরর হ্যান্ডলিং)
  def __init__(self, db_path='memory.db'):
      try:
          self.conn = sqlite3.connect(db_path, check_same_thread=False)  # WAL মোড চালু concurrency-র জন্য
          self.cursor = self.conn.cursor()
          self.cursor.execute('''CREATE TABLE IF NOT EXISTS memories (id INTEGER PRIMARY KEY, task_id TEXT, content TEXT, timestamp DATETIME, category TEXT)''')
          self.conn.commit()
      except sqlite3.OperationalError as e:
          logging.error(f"DB Error: {e}")
          # রিট্রাই লজিক: tenacity দিয়ে ৩ বার ট্রাই
          @retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(3))
          def retry_connect():
              self.conn = sqlite3.connect(db_path)
          retry_connect()
  
  # ব্যাখ্যা: এই মেথড DB কানেকশন তৈরি করে। মূল অংশ: WAL মোড (check_same_thread=False) concurrency ফিক্স করে। এরর হলে রিট্রাই করে। এটি সেলফ-লার্নিং-এর জন্য ডাটা সেভ করে, যাতে পরবর্তী টাস্কে save_memory() কল করে সিঙ্ক করা যায়।







—--------------------------------------------------------------------------------------------------------------------
















The Mask Personal AI Core – একটা ব্যক্তিগত, সেলফ-আপগ্রেডেবল AI অটোমেশন সিস্টেম, যা তোমার PC এবং ক্লাউডকে হাইব্রিডভাবে ইন্টিগ্রেট করে কাজ করে। এটি প্রথমে একটা সিম্পল চ্যাট-বেসড অটোমেটর হিসেবে শুরু হয়েছে, কিন্তু ভিশন অনুসারে এটি একটা "বুদ্ধিমান" সিস্টেমে পরিণত হবে যা নিজে নিজে শিখে, আপগ্রেড করে, এবং লং-টার্ম টাস্ক ম্যানেজ করে।
পুরনো প্ল্যানে ১০টা ফিচার ছিল (প্লাগিন-ভিত্তিক), কিন্তু নতুন ভিশনে প্লাগিন স্কিপ করে প্রথমে সেলফ-আপগ্রেডেবল বেস তৈরি করা হয়েছে। বর্তমানে সিস্টেমটি Render.com-এ ডেপ্লয়ড এবং GitHub-এ ব্যাকআপড, যা নিচে বিস্তারিত বলছি। আমি সিস্টেমের ভিশন, আর্কিটেকচার, কোর ক্যাপাবিলিটিস, রোডম্যাপ সামারি, এবং বর্তমান ইমপ্লিমেন্টেশন বিস্তারিতভাবে লিখছি।
সিস্টেমের নাম এবং মূল দর্শন
নাম: The Mask Personal AI Core v1.0 (অথবা The Mask Automation Core System, যেমন Render এবং GitHub-এ উল্লেখিত)।
মূল দর্শন: এটি একটা হাইব্রিড, সেলফ-আপগ্রেডেবল, লং-টার্ম মেমরি-সম্পন্ন ব্যক্তিগত AI যা তোমার (অ্যাডমিনের) সবকিছু শেখে, ভুল থেকে শিখে, এবং নিজেকে নিজে আপগ্রেড করে। এটি তোমার PC অটোমেশনের কোর হিসেবে কাজ করবে – চ্যাট থেকে শুরু করে ফাইল ম্যানেজমেন্ট, CMD এক্সিকিউশন, TTS/STT, এবং ভবিষ্যতে হ্যাকিং/PC কন্ট্রোল পর্যন্ত। প্রথম প্রায়োরিটি: সেলফ-আপগ্রেড ক্যাপাবিলিটি, যাতে পরবর্তী ফিচারগুলো নিজে যোগ করতে পারে। এটি "বুদ্ধিমান" হবে মানে, এটি ভুল থেকে লার্ন করে দ্বিতীয়বার একই ভুল করবে না, এবং বছরের পর বছর টাস্ক মনে রাখবে।
আর্কিটেকচার (হাইব্রিড ডিজাইন)
সিস্টেমটি হাইব্রিড – ক্লাউড এবং লোকাল লেয়ারের কম্বিনেশন:
ক্লাউড লেয়ার (Render.com):
মডেল: 7B প্যারামিটার LLM (যেমন Llama বা অনুরূপ), Groq 70B ফলব্যাক সহ।
সুবিধা: ১-৫০টা API একসাথে চালানো, স্মার্ট সুইচিং (লিমিট শেষ হলে অটো সুইচ), কম খরচে স্কেলেবল।
ব্যবহার: লাইটওয়েট টাস্ক, চ্যাট, এবং ফাস্ট রেসপন্স।
লোকাল লেয়ার (তোমার PC):
মডেল: 14B প্যারামিটার LLM (Ollama দিয়ে রান)।
সুবিধা: ভারী কাজ (e.g., পার্সোনাল ফাইল অ্যানালাইসিস, লং-টার্ম মেমরি প্রসেসিং), প্রাইভেসি (ডাটা লোকাল থাকে), কোনো API লিমিট নেই।
ব্যবহার: সেনসিটিভ ডাটা, অফলাইন মোড।
স্মার্ট সুইচিং ইঞ্জিন:
কাজের ধরন (e.g., সিম্পল চ্যাট vs কম্প্লেক্স কোডিং), খরচ, স্পিড, API লিমিট, অ্যাভেলেবিলিটি অনুসারে অটো সিদ্ধান্ত নেয়।
উদাহরণ: API লিমিট শেষ হলে লোকালে সুইচ, অথবা সেনসিটিভ টাস্ক লোকালে।
ডাটা স্টোরেজ এবং সিঙ্ক: SQLite ডাটাবেস (প্রাইমারি) + JSON ব্যাকআপ। ক্লাউড এবং লোকালের মধ্যে অটো সিঙ্ক্রোনাইজেশন (মেমরি শেয়ার)।
সিকিউরিটি লেয়ার: অ্যাপ্রুভাল সিস্টেম (সেনসিটিভ অপারেশনের আগে ইউজার অনুমতি), এনক্রিপশন (সেনসিটিভ ডাটা), রুল-বেসড চেকস।
কোর ক্যাপাবিলিটিস (প্রথমে তৈরি করা ফিচারস)
সিস্টেমের মূল ক্ষমতা নিচেরগুলো, যা রোডম্যাপ অনুসারে ধাপে ধাপে যোগ হবে:
যেকোনো ফিচার অনুরোধ অ্যানালাইজ: ইউজারের কমান্ড (e.g., "TTS যোগ করো") অ্যানালাইজ করে কোড জেনারেট এবং আপগ্রেড।
সিস্টেম অ্যাক্সেস: অনুমতি সাপেক্ষে ফাইল তৈরি/এডিট/ডিলিট, CMD/PowerShell এক্সিকিউশন।
কোডিং দক্ষতা: এরর অটো ফিক্স, কোড জেনারেশন।
সেলফ-আপগ্রেড: নিজে কোড লিখে, টেস্ট করে, git push, Render রিস্টার্ট। ব্যাকআপ (zip + Google Drive), রোলব্যাক মেকানিজম সহ।
লং-টার্ম মেমরি: টাস্ক, ইন্টার্যাকশন, লার্নিং লেসন সেভ (বছরের পর বছর)। অটো সামারাইজেশন পুরনো মেমরির।
সেলফ-লার্নিং: ভুল থেকে লেসন সেভ, হাইব্রিড লার্নিং (লোকাল + Groq)।
অ্যাডভান্সড ফিচারস (পরবর্তীতে): TTS/STT (ভয়েস ইন্টারফেস), প্লাগিন সিস্টেম, ড্যাশবোর্ড UI (Streamlit), লং-টার্ম টাস্ক ম্যানেজমেন্ট।
এরর হ্যান্ডলিং এবং মনিটরিং: লগিং, অটো-লার্নিং, /status এন্ডপয়েন্ট (সব মডেলের স্টেট)।
রোডম্যাপ সামারি (৩টা ফেজ)
ফেজ ১: কোর ফাউন্ডেশন + হাইব্রিড + সিকিউরিটি (৪-৫ সপ্তাহ): লং-টার্ম মেমরি ফিক্স, হাইব্রিড সুইচিং, অ্যাপ্রুভাল সিস্টেম, লোকাল কানেকশন, সিঙ্ক্রোনাইজেশন। লক্ষ্য: স্থিতিশীল বেস।
ফেজ ২: সেলফ-আপগ্রেড ইঞ্জিন (৪-৬ সপ্তাহ): SelfUpgradeEngine, কোড ফ্লো, ব্যাকআপ/রোলব্যাক, অটো প্রস্তাব, টেস্টস। লক্ষ্য: নিজে আপগ্রেড করার ক্ষমতা।
ফেজ ৩: অ্যাডভান্সড লার্নিং + লং-টার্ম টাস্ক (৩-৪ সপ্তাহ): SelfLearningManager, টাস্ক স্টেট, হাইব্রিড লার্নিং, সামারাইজেশন, সিকিউরিটি অডিট, প্লাগিন বেস, ভয়েস ইন্টারফেস, ড্যাশবোর্ড, E2E টেস্ট, v1.0 রিলিজ। লক্ষ্য: সত্যিকারের বুদ্ধিমান সিস্টেম।
সময়কাল: মোট ১১-১৫ সপ্তাহ, ছোট টাস্কে ভাগ করা।
বর্তমান তৈরিকৃত ইনফরমেশন এবং স্ট্যাটাস
তোমার প্রদত্ত লিঙ্কগুলো অ্যানালাইজ করে (Render.com সার্ভিস এবং GitHub রেপো), বর্তমান সিস্টেমের অবস্থা নিচের মতো:
Render.com সার্ভিস:
Service ID: srv-d6309dcr85hc739uvh0g
Service Address: https://the-mask-automation-core.onrender.com
বর্তমান ফাংশনালিটি: সার্ভিসটি চালু আছে এবং একটা সিম্পল JSON রেসপন্স দেয়: {"message": "The Mask Core System চালু আছে! বাংলায় কথা বলতে পারি।"}। এটি কোর চ্যাট ফিচারের ইন্ডিকেটর – সিস্টেম অ্যাকটিভ এবং বাংলায় ইন্টার্যাকশন সমর্থন করে। কোনো অতিরিক্ত এন্ডপয়েন্ট (e.g., /status, /chat) দৃশ্যমান নয়, কিন্তু এটি একটা API-ভিত্তিক সার্ভিস বলে মনে হয় যা চ্যাট বা অটোমেশনের বেস। লং-টার্ম মেমরি বা অন্য ফিচারস এখনো পাবলিকলি এক্সপোজড নয় (connection refused এররের কথা মনে করে, এটি আংশিক)।
স্ট্যাটাস: ✅ কমপ্লিট (কোর চ্যাট লাইভ), কিন্তু লং-টার্ম মেমরি ❌ (এখনো ফিক্স নয়)।
GitHub রেপো:
URL: https://github.com/The-Mask-Of-Imran/The-Mask-Core-System
ডেসক্রিপশন: "My PC Automator Core System Backup" – এটি তোমার PC অটোমেটরের কোর সিস্টেমের ব্যাকআপ।
ফাইলস এবং স্ট্রাকচার:
app.py: মূল অ্যাপ্লিকেশন স্ক্রিপ্ট (সম্ভবত Flask/FastAPI-ভিত্তিক, চ্যাট লজিক এখানে)।
config.json: কনফিগারেশন ফাইল (API কী, মডেল সেটিংস, ইত্যাদি)।
memory.json: লং-টার্ম মেমরি স্টোরেজ (JSON-ভিত্তিক ব্যাকআপ, কিন্তু এখনো অস্থিতিশীল)।
requirements.txt: ডিপেন্ডেন্সিস লিস্ট (Python লাইব্রেরী যেমন requests, ollama, ইত্যাদি)।
কমিটস: মোট ৬টা কমিট (সাম্প্রতিক পরিবর্তনগুলো সিস্টেমের বেসিক সেটআপ)।
README: নেই (পরে যোগ করা যাবে ডকুমেন্টেশনের জন্য)।
স্ট্যাটাস: রেপোটি ব্যাকআপ হিসেবে কাজ করছে, কিন্তু পুরো রোডম্যাপের শুধুমাত্র কোর চ্যাট কমপ্লিট। বাকি টাস্কস (e.g., মেমরি ফিক্স, সেলফ-আপগ্রেড) ইমপ্লিমেন্ট করা বাকি।
সামগ্রিক বর্তমান স্ট্যাটাস (মেসেজ থেকে):
✅ কোর চ্যাট (ফিচার ১) কমপ্লিট এবং লাইভ (Render-এ)।
❌ লং-টার্ম মেমরি (ফিচার ২) – Connection refused এরর, ফিক্স দরকার।
বাকি সব ❌ (e.g., সেলফ-আপগ্রেড, হাইব্রিড সুইচিং)।
সমস্যা: মেমরি সিস্টেম অস্থিতিশীল, যা সেলফ-আপগ্রেডের জন্য আবশ্যক।

বিল্ডাপ রোডম্যাপ

### টাস্ক ১
* **ফিচারের নাম**: Long-Term Memory Fix (Connection Refused Error Resolution with SQLite and JSON Backup)
* **ফিচারের বিস্তারিত বর্ণনা**: এই ফিচারটি সিস্টেমের লং-টার্ম মেমরি সিস্টেমকে স্থিতিশীল করে, যাতে কানেকশন রিফিউজড এরর না আসে। এটি টাস্ক, ইউজার ইন্টার্যাকশন, লার্নিং লেসন এবং আপগ্রেড ইতিহাসকে বছরের পর বছর সেভ করে রাখবে। SQLite ডাটাবেসকে প্রাইমারি স্টোরেজ হিসেবে ব্যবহার করা হবে, এবং JSON ফাইলকে ব্যাকআপ হিসেবে, যাতে ডাটা লস না হয়। TaskMemoryManager ক্লাসকে আপডেট করে মেমরি অ্যাক্সেস, রিট্রিভাল এবং আপডেট প্রক্রিয়াকে স্ট্রিমলাইন করা হবে, যাতে সেলফ-আপগ্রেড এবং লং-টার্ম লার্নিং সমর্থন করে।
* **ফিচারটি কিভাবে ইমপ্লিমেন্ট করা হবে**: 
  - **লজিক**: প্রথমে কানেকশন রিফিউজড এররের রুট কজ ডায়াগনোজ করা (যেমন: পোর্ট কনফ্লিক্ট, ফায়ারওয়াল ইস্যু, বা ডাটাবেস লকিং)। TaskMemoryManager ক্লাসে একটা init মেথড যোগ করা যাবে যা ডাটাবেস কানেকশন ট্রাই করবে, ফেল হলে রিট্রাই মেকানিজম (exponential backoff) চালাবে। ডাটা স্টোরেজের জন্য SQLite টেবল তৈরি করা: একটা টেবল 'memories' সাথে কলামগুলো id (PRIMARY KEY), task_id, content (JSON string), timestamp, category (e.g., 'task', 'learning', 'upgrade')। প্রত্যেক অপারেশনের পর JSON ফাইলে (e.g., memory_backup.json) ডাম্প করা। রিট্রিভাল লজিক: query_by_category অথবা query_by_time মেথড যোগ। এরর হ্যান্ডলিং: try-except ব্লক দিয়ে sqlite3.OperationalError ক্যাচ করে লগ করা এবং অটো-রিকভারি (e.g., ডাটাবেস রি-ইনিশিয়ালাইজ)।
  - **লাইব্রেরী এবং টুলস**: Python's sqlite3 (built-in for database), json (built-in for backup), logging (for error logs), time/retries লাইব্রেরী যেমন tenacity (যদি pip install সম্ভব হয়, নইলে ম্যানুয়াল retry loop)। Ollama অথবা Render এনভায়রনমেন্টে ইন্টিগ্রেট করার জন্য os.environ দিয়ে DB_PATH সেট করা।
  - **স্টেপ-বাই-স্টেপ ইমপ্লিমেন্টেশন**: 1. TaskMemoryManager.py ফাইলে ক্লাস আপডেট। 2. __init__-এ DB কানেকশন তৈরি। 3. save_memory, load_memory মেথড যোগ। 4. backup_to_json মেথড। 5. টেস্ট: একটা স্যাম্পল মেমরি সেভ করে লোড করা এবং এরর সিমুলেট করে চেক।
* **ফিচারের অসুবিধা বা কমতি**: SQLite-এ ডাটা করাপশন হতে পারে যদি একাধিক প্রসেস একসাথে অ্যাক্সেস করে (concurrency issue), পারফরম্যান্স স্লো হতে পারে বড় ডাটাসেটে (e.g., হাজার হাজার এন্ট্রি), JSON ব্যাকআপ ম্যানুয়ালি ম্যানেজ করতে হবে যা স্কেল না করে, এবং সিকিউরিটি রিস্ক যদি ডাটাবেস এনক্রিপ্ট না হয় (সেনসিটিভ ইউজার ডাটা লিক হতে পারে)।
* **ফিচারের অসুবিধা দূর করার জন্য কি ব্যবস্থা গ্রহণ করা যায়**: Concurrency-র জন্য WAL (Write-Ahead Logging) মোড চালু করা (sqlite3.connect(db, check_same_thread=False))। পারফরম্যান্সের জন্য ইন্ডেক্সিং যোগ (e.g., CREATE INDEX on timestamp) এবং পিরিয়ডিক ক্লিনআপ (old memories archive)। JSON ব্যাকআপ অটোমেট করার জন্য cron job অথবা scheduler লাইব্রেরী (e.g., schedule.py) ব্যবহার। সিকিউরিটির জন্য sqlcipher (encrypted SQLite) ইউজ করা অথবা keyring লাইব্রেরী দিয়ে পাসওয়ার্ড প্রোটেকশন যোগ।

### টাস্ক ২
* **ফিচারের নাম**: Hybrid Switching Engine (ModelRouter Class)
* **ফিচারের বিস্তারিত বর্ণনা**: এই ফিচারটি একটা স্মার্ট রাউটার তৈরি করবে যা ক্লাউড (Render.com-এ 7B মডেল + Groq 70B ফলব্যাক) এবং লোকাল (PC-এ 14B মডেল) লেয়ারের মধ্যে অটোমেটিক সুইচ করবে। এটি টাস্কের ধরন (e.g., সিম্পল চ্যাট vs ভারী কম্পুটেশন), API লিমিট, খরচ, স্পিড এবং অ্যাভেলেবিলিটি অনুসারে সিদ্ধান্ত নেবে, যাতে সিস্টেম সবসময় অপটিমাইজড এবং রিলায়েবল থাকে।
* **ফিচারটি কিভাবে ইমপ্লিমেন্ট করা হবে**: 
  - **লজিক**: ModelRouter ক্লাসে একটা route_request মেথড যোগ, যা ইনপুট প্যারামিটার (task_type, urgency, data_size) অ্যানালাইজ করে স্কোর ক্যালকুলেট করবে (e.g., if data_size > threshold, use local; if API limit exceeded, fallback to local)। প্রায়োরিটি রুলস: 1. চেক অ্যাভেলেবিলিটি (ping অথবা health check), 2. লোড ব্যালেন্স (round-robin if multiple APIs), 3. ফেলওভার (if cloud down, switch to local)। লগিং: প্রত্যেক সুইচ লগ করা।
  - **লাইব্রেরী এবং টুলস**: requests (for API calls to Groq/Render), ollama (for local model integration), os/time (for health checks), logging। Groq API key env var দিয়ে সেট করা, Render endpoint URL হার্ডকোড অথবা config ফাইলে।
  - **স্টেপ-বাই-স্টেপ ইমপ্লিমেন্টেশন**: 1. ModelRouter.py ফাইল তৈরি। 2. __init__-এ মডেল কনফিগ লোড (e.g., dict with 'cloud': {'model': '7B', 'api': 'groq'}, 'local': {'model': '14B'}). 3. route_request মেথড ইমপ্লিমেন্ট। 4. টেস্ট: স্যাম্পল কোয়েরি পাঠিয়ে সুইচ চেক।
* **ফিচারের অসুবিধা বা কমতি**: লেটেন্সি বাড়তে পারে সুইচিং ডিসিশন নিতে (decision overhead), লোকাল মডেল অ্যাভেলেবল না থাকলে (PC off) ফেল হবে, খরচ ট্র্যাকিং অ্যাকুরেট না হলে ওভারস্পেন্ডিং, এবং কমপ্লেক্স টাস্কে ভুল রাউটিং (e.g., sensitive data to cloud)।
* **ফিচারের অসুবিধা দূর করার জন্য কি ব্যবস্থা গ্রহণ করা যায়**: Decision overhead কমাতে caching (e.g., recent routes cache) যোগ। লোকাল অ্যাভেলেবিলিটির জন্য heartbeat mechanism (periodic ping from local to cloud)। খরচ ট্র্যাকিংয়ের জন্য prometheus অথবা simple counter ইউজ। রাউটিং ভুলের জন্য ML-based predictor (future upgrade) অথবা rule-based overrides (e.g., sensitive: always local)।

### টাস্ক ৩
* **ফিচারের নাম**: API Limit Monitoring + Smart Switching Logic
* **ফিচারের বিস্তারিত বর্ণনা**: এই ফিচারটি Groq/Render API-এর রেট লিমিট (e.g., requests per minute) মনিটর করবে এবং লিমিট শেষ হলে অটোমেটিক সুইচ করবে অন্য মডেলে (e.g., local 14B)। স্মার্ট লজিক যোগ করে প্রিডিক্টিভ সুইচিং করবে (e.g., limit close হলে early switch), যাতে ডাউনটাইম না হয় এবং খরচ অপটিমাইজ হয়।
* **ফিচারটি কিভাবে ইমপ্লিমেন্ট করা হবে**: 
  - **লজিক**: একটা LimitMonitor ক্লাস তৈরি যা API calls ট্র্যাক করবে (counter with timestamp), remaining limit ক্যালকুলেট (from API headers like X-RateLimit-Remaining)। সুইচিং লজিক: if remaining < threshold (e.g., 10%), trigger switch। প্রিডিক্টিভ: historical usage দিয়ে forecast (simple moving average)।
  - **লাইব্রেরী এবং টুলস**: requests (API headers parse), collections.deque (for sliding window counters), threading (background monitoring), logging। Groq SDK যদি available, তাহলে ব্যবহার।
  - **স্টেপ-বাই-স্টেপ ইমপ্লিমেন্টেশন**: 1. ModelRouter-এ integrate। 2. post_api_call মেথড যোগ headers update-এর জন্য। 3. monitor_loop background thread। 4. টেস্ট: simulate limits exceeding।
* **ফিচারের অসুবিধা বা কমতি**: API headers inaccurate হলে ট্র্যাকিং ভুল, background monitoring resource consume করতে পারে (CPU usage), প্রিডিকশন ভুল হলে unnecessary switches, এবং multi-API (1-50) ম্যানেজ করা কমপ্লেক্স।
* **ফিচারের অসুবিধা দূর করার জন্য কি ব্যবস্থা গ্রহণ করা যায়**: Headers inaccuracy-র জন্য manual fallback counter। Resource usage কমাতে lightweight asyncio ইউজ। প্রিডিকশন ইম্প্রুভের জন্য ML library (e.g., scikit-learn for better forecasting)। Multi-API-র জন্য pool manager (e.g., concurrent.futures)।

### টাস্ক ৪
* **ফিচারের নাম**: Approval System (ApprovalManager - First Version)
* **ফিচারের বিস্তারিত বর্ণনা**: এই ফিচারটি সেনসিটিভ অপারেশন (e.g., file delete, code upgrade) এর আগে ইউজার অ্যাপ্রুভাল নেবে, যাতে সিকিউরিটি মেইনটেইন হয়। প্রথম ভার্সনে সিম্পল UI (console prompt অথবা API endpoint) দিয়ে yes/no অপশন, এবং লগিং সব অ্যাপ্রুভালের।
* **ফিচারটি কিভাবে ইমপ্লিমেন্ট করা হবে**: 
  - **লজিক**: ApprovalManager ক্লাসে request_approval মেথড যোগ, যা action description পাঠিয়ে ইউজারকে প্রম্পট করবে (e.g., "Approve file delete? Y/N")। Timeout if no response। Rule-based: certain actions auto-approve if low risk।
  - **লাইব্রেরী এবং টুলস**: input() for console, flask/fastapi for API-based UI, logging। Future: telegram bot for notifications।
  - **স্টেপ-বাই-স্টেপ ইমপ্লিমেন্টেশন**: 1. ApprovalManager.py তৈরি। 2. integrate with sensitive modules। 3. টেস্ট: mock actions।
* **ফিচারের অসুবিধা বা কমতি**: User interaction delay (real-time tasks slow), console UI not user-friendly, spoofing risk if not authenticated, এবং auto-approve rules incomplete হলে security hole।
* **ফিচারের অসুবিধা দূর করার জন্য কি ব্যবস্থা গ্রহণ করা যায়**: Delay কমাতে async approvals (e.g., queue actions)। UI ইম্প্রুভের জন্য Streamlit dashboard। Authentication-এর জন্য JWT tokens। Rules refine করতে ML-based risk assessment (future)।

### টাস্ক ৫
* **ফিচারের নাম**: Basic File + CMD Execution Module (Permission-Based)
* **ফিচারের বিস্তারিত বর্ণনা**: এই ফিচারটি সিস্টেমকে ফাইল অপারেশন (create/delete/edit) এবং CMD/PowerShell কমান্ড এক্সিকিউট করার ক্ষমতা দেবে, কিন্তু শুধু ইউজার অ্যাপ্রুভাল সাপেক্ষে। এটি সেলফ-আপগ্রেডের বেস হবে, যেমন কোড ফাইল এডিট।
* **ফিচারটি কিভাবে ইমপ্লিমেন্ট করা হবে**: 
  - **লজিক**: ExecutionModule ক্লাসে execute_cmd এবং file_op মেথড যোগ, যা approval check করে তারপর subprocess.call অথবা os.open ব্যবহার। Sandboxing: restricted dirs (e.g., only project folder)। Output capture এবং error handling।
  - **লাইব্রেরী এবং টুলস**: subprocess (for CMD), os/shutil (for files), logging। Windows-specific: powershell.exe।
  - **স্টেপ-বাই-স্টেপ ইমপ্লিমেন্টেশন**: 1. Module তৈরি। 2. Integrate with ApprovalManager। 3. টেস্ট: safe commands like 'echo test'।
* **ফিচারের অসুবিধা বা কমতি**: Security risk (malicious commands), platform dependency (Windows vs Linux), output parsing complex, এবং long-running commands block system।
* **ফিচারের অসুবিধা দূর করার জন্য কি ব্যবস্থা গ্রহণ করা যায়**: Risk mitigate করতে whitelist commands। Platform independence-এর জন্য cross-platform libs (e.g., shlex)। Parsing-এর জন্য regex। Async execution with threading/asyncio।
### টাস্ক ৬
* **ফিচারের নাম**: Local 14B Model Connection (Ollama Integration)
* **ফিচারের বিস্তারিত বর্ণনা**: এই ফিচারটি লোকাল PC-এ চলা 14B প্যারামিটার মডেলকে সিস্টেমের সাথে কানেক্ট করবে Ollama ফ্রেমওয়ার্ক ব্যবহার করে। এটি ভারী কম্পুটেশনাল টাস্ক (যেমন: লং-টার্ম মেমরি প্রসেসিং, পার্সোনাল ফাইল অ্যানালাইসিস) এর জন্য লোকাল মডেলকে ব্যবহার করবে, যাতে প্রাইভেসি মেইনটেইন হয় এবং ক্লাউড খরচ কমে। কানেকশন স্টেবল হবে, অটো-রিকানেক্ট মেকানিজম সহ, এবং মডেলের স্টেট (রানিং/আইডল) মনিটর করবে।
* **ফিচারটি কিভাবে ইমপ্লিমেন্ট করা হবে**: 
  - **লজিক**: Ollama API-কে ইউজ করে একটা LocalModelConnector ক্লাস তৈরি করা, যা init-এ Ollama সার্ভার চেক করবে (e.g., if not running, auto-start via subprocess)। generate_response মেথড যোগ করে প্রম্পট পাঠানো এবং রেসপন্স রিসিভ করা। এরর হ্যান্ডলিং: ConnectionError ক্যাচ করে retry (e.g., 3 attempts)। ইন্টিগ্রেশন: ModelRouter-এ লোকাল অপশন যোগ করে সুইচ করা।
  - **লাইব্রেরী এবং টুলস**: ollama (pip install ollama), requests (API calls-এর জন্য, যদি Ollama HTTP endpoint ব্যবহার হয়), subprocess (Ollama start/stop-এর জন্য), logging (স্টেট লগিং)। Ollama config: model='llama-14b' সেট করা env var দিয়ে।
  - **স্টেপ-বাই-স্টেপ ইমপ্লিমেন্টেশন**: 1. Ollama ইনস্টল এবং 14B মডেল ডাউনলোড চেক। 2. LocalModelConnector.py তৈরি। 3. connect() মেথড ইমপ্লিমেন্ট। 4. টেস্ট: স্যাম্পল প্রম্পট পাঠিয়ে রেসপন্স চেক।
* **ফিচারের অসুবিধা বা কমতি**: লোকাল মডেল চালাতে হাই GPU/CPU রিসোর্স লাগবে (e.g., 16GB+ RAM), যা লো-এন্ড PC-এ স্লো হতে পারে; Ollama সার্ভার ডাউন হলে কানেকশন ফেল; প্রাইভেসি ইস্যু যদি লোকাল ডাটা শেয়ার হয়; এবং প্ল্যাটফর্ম ডিপেন্ডেন্সি (Windows/Mac/Linux-এ Ollama সাপোর্ট ভিন্ন)।
* **ফিচারের অসুবিধা দূর করার জন্য কি ব্যবস্থা গ্রহণ করা যায়**: রিসোর্স ইস্যুর জন্য quantized মডেল (e.g., 14B-Q4) ইউজ করা অথবা fallback to smaller local model। ডাউনটাইমের জন্য watchdog thread (periodic health check) যোগ। প্রাইভেসির জন্য data encryption (e.g., cryptography lib)। প্ল্যাটফর্মের জন্য cross-platform checks (os.platform) এবং conditional code।

### টাস্ক ৭
* **ফিচারের নাম**: Smart Switching Test (Cloud vs Local)
* **ফিচারের বিস্তারিত বর্ণনা**: এই ফিচারটি হাইব্রিড সুইচিং ইঞ্জিনকে টেস্ট করবে, যাতে ক্লাউড (7B/Groq 70B) এবং লোকাল (14B) মডেলের মধ্যে সুইচিং সঠিকভাবে কাজ করে কিনা চেক হয়। টেস্ট কেসগুলোতে লোড, লিমিট, স্পিড, এবং এরর সিনারিও কভার করা হবে, যাতে সিস্টেমের রিলায়েবিলিটি নিশ্চিত হয়। রিপোর্ট জেনারেট করবে সাকসেস রেট, টাইমিং, এবং ফেলিউর কজ সহ।
* **ফিচারটি কিভাবে ইমপ্লিমেন্ট করা হবে**: 
  - **লজিক**: একটা TestSuite ক্লাস তৈরি যা মাল্টিপল টেস্ট কেস রান করবে (e.g., high-load: 10 concurrent requests; limit-exceed: simulate API limit; offline: disconnect local)। প্রত্যেক টেস্টে ModelRouter.route_request কল করে রেসপন্স চেক (correct model used? response valid?)। মেট্রিক্স: time_taken, success (assert), error_type লগ করা।
  - **লাইব্রেরী এবং টুলস**: unittest/pytest (testing framework), mock (simulate errors/limits), time/performance (timing), logging/json (report generation)। concurrent.futures (parallel tests)।
  - **স্টেপ-বাই-স্টেপ ইমপ্লিমেন্টেশন**: 1. tests/ ফোল্ডারে switching_test.py তৈরি। 2. Test cases define (e.g., test_cloud_fallback, test_local_priority)। 3. Run tests via command line। 4. Generate report (JSON/HTML)।
* **ফিচারের অসুবিধা বা কমতি**: টেস্টিং এনভায়রনমেন্ট রিয়েল-ওয়ার্ল্ড থেকে ভিন্ন হতে পারে (simulated errors inaccurate), টেস্ট রান করতে সময় লাগবে (long-running), coverage incomplete যদি সব সিনারিও না কভার হয়, এবং dependency on external APIs (Groq downtime during test)।
* **ফিচারের অসুবিধা দূর করার জন্য কি ব্যবস্থা গ্রহণ করা যায়**: Accuracy-র জন্য integration tests with real APIs (but mocked responses)। Time কমাতে selective tests (e.g., --tags)। Coverage ইম্প্রুভের জন্য code coverage tools (coverage.py)। Dependency-র জন্য offline mocks (e.g., responses lib)।

### টাস্ক ৮
* **ফিচারের নাম**: Error Logging + Auto-Learning Basic Framework
* **ফিচারের বিস্তারিত বর্ণনা**: এই ফিচারটি সিস্টেমের সব এররকে লগ করবে (e.g., stack trace, context) এবং একটা বেসিক অটো-লার্নিং মেকানিজম তৈরি করবে যা এরর থেকে লেসন সেভ করে (লং-টার্ম মেমরিতে) এবং ফিউচার অপারেশনে অ্যাভয়েড করবে (e.g., known error patterns skip)। এটি সেলফ-ইম্প্রুভমেন্টের বেস হবে, যাতে সিস্টেম ভুল থেকে শিখে।
* **ফিচারটি কিভাবে ইমপ্লিমেন্ট করা হবে**: 
  - **লজিক**: ErrorLogger ক্লাস তৈরি যা try-except wrappers দিয়ে এরর ক্যাচ করে log_to_file (timestamp, error_type, message, stack)। AutoLearning: post-error, analyze_error মেথড যা pattern match করে (e.g., regex for common errors) এবং lesson generate (e.g., "Avoid X if Y") করে মেমরিতে সেভ। Before action, check_learned_lessons to prevent recurrence।
  - **লাইব্রেরী এবং টুলস**: logging (structured logs), traceback (stack traces), re (regex patterns), json (lesson storage)। Integrate with TaskMemoryManager for saving lessons।
  - **স্টেপ-বাই-স্টেপ ইমপ্লিমেন্টেশন**: 1. ErrorLogger.py তৈরি। 2. Global exception handler যোগ। 3. AutoLearning rules define (dict of patterns)। 4. টেস্ট: simulate errors and check lessons saved।
* **ফিচারের অসুবিধা বা কমতি**: Logging overhead (performance slow if verbose), false positives in pattern matching (wrong lessons), storage bloat if too many logs/lessons, এবং complex errors (e.g., runtime bugs) auto-learn না হওয়া।
* **ফিচারের অসুবিধা দূর করার জন্য কি ব্যবস্থা গ্রহণ করা যায়**: Overhead কমাতে log levels (DEBUG/INFO/ERROR)। Accuracy-র জন্য ML-based classification (future, e.g., scikit-learn)। Storage-এর জন্য periodic cleanup (e.g., rotate logs)। Complex errors-এর জন্য user feedback loop (approve lessons)।

### টাস্ক ৯
* **ফিচারের নাম**: /status Endpoint Upgrade (Show All Models' State)
* **ফিচারের বিস্তারিত বর্ণনা**: এই ফিচারটি Render-এর API-এ /status এন্ডপয়েন্টকে আপগ্রেড করবে, যাতে সব মডেলের স্টেট (cloud 7B, Groq 70B, local 14B) দেখাবে: running/idle/down, usage stats (requests, limits), health (ping response), এবং overall system health। এটি মনিটরিং এবং ডিবাগিং সহজ করবে।
* **ফিচারটি কিভাবে ইমপ্লিমেন্ট করা হবে**: 
  - **লজিক**: Flask/FastAPI app-এ @app.route('/status') যোগ করে JSON response generate: dict with 'models': {'cloud_7b': {'status': 'running', 'uptime': X}, ...}। Health check: ping each model (e.g., small prompt test)। Stats from LimitMonitor integrate।
  - **লাইব্রেরী এবং টুলস**: flask/fastapi (API framework), requests (health pings), json (response), time (uptime calc)। Deploy on Render with env vars।
  - **স্টেপ-বাই-স্টেপ ইমপ্লিমেন্টেশন**: 1. main.py-এ endpoint update। 2. get_model_status helper function। 3. Integrate with ModelRouter। 4. টেস্ট: curl /status to check output।
* **ফিচারের অসুবিধা বা কমতি**: Security risk if endpoint public (expose sensitive stats), performance hit if frequent calls, incomplete stats if monitors not synced, এবং scalability issue for more models।
* **ফিচারের অসুবিধা দূর করার জন্য কি ব্যবস্থা গ্রহণ করা যায়**: Security-র জন্য auth (API key required)। Perf-এর জন্য caching (e.g., redis, but simple dict cache)। Sync-এর জন্য real-time updates (websockets future)। Scalability-এর জন্য modular design (add_model method)।

### টাস্ক ১০
* **ফিচারের নাম**: Render + Local Synchronization (Memory Share)
* **ফিচারের বিস্তারিত বর্ণনা**: এই ফিচারটি ক্লাউড (Render) এবং লোকাল (PC) লেয়ারের মধ্যে লং-টার্ম মেমরি শেয়ার করবে, যাতে ডাটা সিঙ্ক থাকে (e.g., new memory in cloud auto-sync to local)। এটি bidirectional sync সাপোর্ট করবে, conflict resolution (e.g., timestamp-based), এবং offline mode (queue changes)।
* **ফিচারের বিস্তারিত বর্ণনা**: SyncEngine ক্লাস তৈরি যা periodic sync (e.g., every 5min) রান করবে: pull/push memories via API (local exposes endpoint)। Conflict: latest timestamp wins। Offline: queue in local file, sync on connect।
  - **লাইব্রেরী এবং টুলস**: requests (API sync), sqlite3/json (data format), schedule/threading (periodic tasks), logging (sync logs)। Optional: rsync for file-based if DB shared।
  - **স্টেপ-বাই-স্টেপ ইমপ্লিমেন্টেশন**: 1. SyncEngine.py তৈরি। 2. Local endpoint for /sync। 3. Integrate with TaskMemoryManager। 4. টেস্ট: create memory in one, check in other।
* **ফিচারের অসুবিধা বা কমতি**: Network latency/downtime cause sync failures, data conflicts lead to loss, security (data in transit), এবং resource use for frequent syncs।
* **ফিচারের অসুবিধা দূর করার জন্য কি ব্যবস্থা গ্রহণ করা যায়**: Failures-এর জন্য retry queue। Conflicts-এর জন্য merge logic (e.g., diff tools)। Security-এর জন্য HTTPS + encryption (ssl lib)। Resource-এর জন্য adaptive sync (based on changes)।
### টাস্ক ১১
* **ফিচারের নাম**: SelfUpgradeEngine Class Creation
* **ফিচারের বিস্তারিত বর্ণনা**: এই ফিচারটি একটা কেন্দ্রীয় ক্লাস তৈরি করবে যা সিস্টেমের সেলফ-আপগ্রেড প্রক্রিয়াকে ম্যানেজ করবে। SelfUpgradeEngine ক্লাসটি ইউজারের অনুরোধ (e.g., নতুন ফিচার যোগ) অ্যানালাইজ করে কোড জেনারেট করবে, টেস্ট করবে, এবং ডেপ্লয় করবে। এটি অ্যাপ্রুভাল সিস্টেম, মেমরি ম্যানেজার, এবং এক্সিকিউশন মডিউলের সাথে ইন্টিগ্রেট হবে, যাতে সিস্টেম নিজেকে নিরাপদে এবং অটোমেটিকভাবে আপগ্রেড করতে পারে।
* **ফিচারটি কিভাবে ইমপ্লিমেন্ট করা হবে**: 
  - **লজিক**: ক্লাসে init মেথড যোগ করে কনফিগ লোড (e.g., git repo, model router)। upgrade_request মেথড যা প্রম্পট থেকে কোড জেনারেট (using LLM like Groq), validate_code (syntax check), এবং execute_upgrade (step-by-step flow)। ইন্টিগ্রেশন: ApprovalManager থেকে permission নেয়া এবং ErrorLogger দিয়ে হ্যান্ডেল।
  - **লাইব্রেরী এবং টুলস**: ast (code syntax validation), gitpython (git operations), requests (LLM API calls), logging। Ollama/Groq for code generation prompts।
  - **স্টেপ-বাই-স্টেপ ইমপ্লিমেন্টেশন**: 1. SelfUpgradeEngine.py ফাইল তৈরি। 2. __init__ এবং core methods যোগ। 3. Prompt template define (e.g., "Generate code for feature X")। 4. টেস্ট: dummy upgrade request simulate।
* **ফিচারের অসুবিধা বা কমতি**: LLM-generated code buggy হতে পারে (syntax/infinite loops), resource intensive (code gen + test), security risk (malicious code injection), এবং dependency on external APIs (if Groq down)।
* **ফিচারের অসুবিধা দূর করার জন্য কি ব্যবস্থা গ্রহণ করা যায়**: Buggy code-এর জন্য multi-stage validation (ast.parse + pylint)। Resource-এর জন্য async processing (asyncio)। Security-এর জন্য sandbox execution (restricted env)। Dependency-এর জন্য local fallback model।

### টাস্ক ১২
* **ফিচারের নাম**: Code Generate → Local Test → Git Push → Render Restart Flow
* **ফিচারের বিস্তারিত বর্ণনা**: এই ফিচারটি সেলফ-আপগ্রেডের সম্পূর্ণ ফ্লো তৈরি করবে: LLM দিয়ে কোড জেনারেট, লোকালে টেস্ট (unit/integration), git-এ পুশ, এবং Render-এ অটো রিস্টার্ট। এটি পাইপলাইন হিসেবে চলবে, যাতে ম্যানুয়াল ইন্টারভেনশন ছাড়াই আপগ্রেড হয়, এবং লগিং সব স্টেপের।
* **ফিচারটি কিভাবে ইমপ্লিমেন্ট করা হবে**: 
  - **লজিক**: SelfUpgradeEngine-এ execute_flow মেথড যোগ: 1. generate_code (prompt to LLM), 2. test_locally (subprocess.run tests), 3. git_push (commit + push), 4. render_restart (API call to Render webhook)। If any step fails, abort and log।
  - **লাইব্রেরী এবং টুলস**: gitpython (git ops), subprocess (test running), requests (Render API), unittest/pytest (testing)। Env vars for git creds/Render token।
  - **স্টেপ-বাই-স্টেপ ইমপ্লিমেন্টেশন**: 1. Flow method define। 2. Each sub-method implement। 3. Webhook setup in Render। 4. টেস্ট: end-to-end dummy code change।
* **ফিচারের অসুবিধা বা কমতি**: Test failures frequent (incomplete tests), git conflicts (concurrent changes), Render restart downtime, এবং credential exposure risk।
* **ফিচারের অসুবিধা দূর করার জন্য কি ব্যবস্থা গ্রহণ করা যায়**: Failures-এর জন্য comprehensive test suite। Conflicts-এর জন্য auto-rebase। Downtime-এর জন্য blue-green deployment (future)। Creds-এর জন্য secrets manager (e.g., dotenv)।

### টাস্ক ১৩
* **ফিচারের নাম**: Backup System (Zip + Google Drive)
* **ফিচারের বিস্তারিত বর্ণনা**: এই ফিচারটি আপগ্রেডের আগে/পরে সিস্টেমের সম্পূর্ণ ব্যাকআপ নেবে: কোড, ডাটাবেস, কনফিগ ফাইলগুলোকে zip করে Google Drive-এ আপলোড। এটি অটোমেটিক ট্রিগার হবে, ভার্সনিং সাপোর্ট (e.g., timestamped backups) সহ, যাতে ডাটা লস না হয়।
* **ফিচারটি কিভাবে ইমপ্লিমেন্ট করা হবে**: 
  - **লজিক**: BackupManager ক্লাসে create_backup মেথড: shutil.make_archive for zip, then upload_to_drive (using API)। Pre-upgrade trigger in SelfUpgradeEngine। Retention policy: keep last 10 backups।
  - **লাইব্রেরী এবং টুলস**: shutil/zipfile (zipping), google-api-python-client (Drive API), oauth2client (auth)। Setup service account for auth।
  - **স্টেপ-বাই-স্টেপ ইমপ্লিমেন্টেশন**: 1. BackupManager.py তৈরি। 2. Drive API credentials setup। 3. Integrate with upgrade flow। 4. টেস্ট: manual backup and upload।
* **ফিচারের অসুবিধা বা কমতি**: Large backups slow upload, API limits (Drive quota), dependency on internet, এবং privacy risk (sensitive data on cloud)।
* **ফিচারের অসুবিধা দূর করার জন্য কি ব্যবস্থা গ্রহণ করা যায়**: Slow-এর জন্য incremental backups (rsync-like)। Quota-এর জন্য multiple accounts/rotate। Internet-এর জন্য local fallback (e.g., external drive)। Privacy-এর জন্য encryption (cryptography lib) before upload।

### টাস্ক ১৪
* **ফিচারের নাম**: Rollback Mechanism (Git Checkout + Restore)
* **ফিচারের বিস্তারিত বর্ণনা**: এই ফিচারটি আপগ্রেড ফেল হলে অটো/ম্যানুয়াল রোলব্যাক করবে: git checkout previous commit, restore from backup, এবং restart। এটি ইতিহাস ট্র্যাক করবে, যাতে সিস্টেম স্টেবল থাকে এবং ডাউনটাইম কম হয়।
* **ফিচারটি কিভাবে ইমপ্লিমেন্ট করা হবে**: 
  - **লজিক**: RollbackManager ক্লাসে rollback মেথড: git.checkout(prev_commit), restore_backup (unzip + copy files), then restart। Trigger on error detection।
  - **লাইব্রেরী এবং টুলস**: gitpython (checkout), shutil (restore), subprocess (restart script)।
  - **স্টেপ-বাই-স্টেপ ইমপ্লিমেন্টেশন**: 1. RollbackManager.py তৈরি। 2. Get prev commit from git log। 3. Integrate with ErrorLogger। 4. টেস্ট: simulate failed upgrade and rollback।
* **ফিচারের অসুবিধা বা কমতি**: Incomplete rollback (db changes not reverted), git history clutter, manual intervention needed for complex fails, এবং time-consuming for large repos。
* **ফিচারের অসুবিধা দূর করার জন্য কি ব্যবস্থা গ্রহণ করা যায়**: Db-এর জন্য snapshot restore (sqlite backup)। Clutter-এর জন্য squash commits। Complex-এর জন্য user notification। Time-এর জন্য optimized git ops (e.g., shallow clone)।

### টাস্ক ১৫
* **ফিচারের নাম**: Auto Upgrade Proposal + Permission UI
* **ফিচারের বিস্তারিত বর্ণনা**: এই ফিচারটি সিস্টেম নিজে আপগ্রেড প্রস্তাব করবে (e.g., from learned errors) এবং একটা UI (console/web) দিয়ে permission নেবে। প্রস্তাবে details (what, why, risk) থাকবে, যাতে ইউজার ইনফর্মড ডিসিশন নিতে পারে।
* **ফিচারটি কিভাবে ইমপ্লিমেন্ট করা হবে**: 
  - **লজিক**: ProposalGenerator ক্লাসে generate_proposal (from memory analysis), then UI prompt (e.g., Streamlit page or console input)। ApprovalManager integrate।
  - **লাইব্রেরী এবং টুলস**: streamlit (web UI), input() (console), json (proposal format)।
  - **স্টেপ-বাই-স্টেপ ইমপ্লিমেন্টেশন**: 1. ProposalGenerator.py তৈরি। 2. UI script setup। 3. Trigger from AutoLearning। 4. টেস্ট: sample proposal and approve。
* **ফিচারের অসুবিধা বা কমতি**: Proposals irrelevant/frequent (spam), UI not intuitive, async issues (waiting for approval), এবং security (UI spoofing)।
* **ফিচারের অসুবিধা দূর করার জন্য কি ব্যবস্থা গ্রহণ করা যায়**: Irrelevant-এর জন্য threshold scoring। Intuitive-এর জন্য better UX (e.g., buttons)। Async-এর জন্য queue system। Security-এর জন্য auth (password)।

### টাস্ক ১৬
* **ফিচারের নাম**: First Self-Upgrade Test (e.g., Add New /tts Endpoint)
* **ফিচারের বিস্তারিত বর্ণনা**: এই ফিচারটি সেলফ-আপগ্রেড ইঞ্জিনের প্রথম টেস্ট করবে: একটা সিম্পল ফিচার (e.g., /tts endpoint for TTS) যোগ করে চেক করা যাবে পুরো ফ্লো কাজ করে কিনা। রিপোর্ট জেনারেট সাকসেস/ফেল সহ।
* **ফিচারটি কিভাবে ইমপ্লিমেন্ট করা হবে**: 
  - **লজিক**: Test script যা upgrade_request("Add TTS endpoint") কল করে, then verify (e.g., curl /tts)। Metrics: time, errors।
  - **লাইব্রেরী এবং টুলস**: pytest (testing), requests (verification), logging।
  - **স্টেপ-বাই-স্টেপ ইমপ্লিমেন্টেশন**: 1. Test script write। 2. Dummy TTS code template। 3. Run and log results। 4. Adjust based on fails।
* **ফিচারের অসুবিধা বা কমতি**: Test env not real (simulated), incomplete coverage, dependency on prior tasks, এবং manual review needed।
* **ফিচারের অসুবিধা দূর করার জন্য কি ব্যবস্থা গ্রহণ করা যায়**: Real-এর জন্য staging env। Coverage-এর জন্য more cases। Dependency-এর জন্য sequential runs। Review-এর জন্য auto-reports।

### টাস্ক ১৭
* **ফিচারের নাম**: Error Handling Improvement (Auto Rollback on Upgrade Fail)
* **ফিচারের বিস্তারিত বর্ণনা**: এই ফিচারটি আপগ্রেড ফেল হলে অটো রোলব্যাক ট্রিগার করবে, লগ করে, এবং নোটিফাই করবে। ইম্প্রুভড হ্যান্ডলিং: classify errors (recoverable vs fatal) এবং retry for transient।
* **ফিচারটি কিভাবে ইমপ্লিমেন্ট করা হবে**: 
  - **লজিক**: ErrorLogger-এ classify_error, then if fatal, call rollback। Retry loop for transients (e.g., network errors)।
  - **লাইব্রেরী এবং টুলস**: tenacity (retries), logging, smtplib/telegram (notifications)।
  - **স্টেপ-বাই-স্টেপ ইমপ্লিমেন্টেশন**: 1. Error classification rules। 2. Integrate with upgrade flow। 3. Notification setup। 4. টেস্ট: simulate fails।
* **ফিচারের অসুবিধা বা কমতি**: Misclassification (wrong rollback), infinite retries, notification spam, এবং complex error chains।
* **ফিচারের অসুবিধা দূর করার জন্য কি ব্যবস্থা গ্রহণ করা যায়**: Misclass-এর জন্য ML classifier (future)। Retries-এর জন্য max attempts। Spam-এর জন্য throttle। Chains-এর জন্য root cause analysis (traceback)।

### টাস্ক ১৮
* **ফিচারের নাম**: Save Upgrade History in Long-Term Memory
* **ফিচারের বিস্তারিত বর্ণনা**: এই ফিচারটি প্রত্যেক আপগ্রেডের ইতিহাস (what changed, when, success/fail) লং-টার্ম মেমরিতে সেভ করবে, যাতে ফিউচার লার্নিং/অডিট সম্ভব হয়। Queryable format (e.g., by date/feature)।
* **ফিচারটি কিভাবে ইমপ্লিমেন্ট করা হবে**: 
  - **লজিক**: Post-upgrade, save_to_memory (JSON dict: {'upgrade_id', 'changes', 'status'}) using TaskMemoryManager।
  - **লাইব্রেরী এবং টুলস**: json, sqlite3 (via MemoryManager)।
  - **স্টেপ-বাই-স্টেপ ইমপ্লিমেন্টেশন**: 1. History schema in DB। 2. Hook in upgrade flow। 3. Query methods add। 4. টেস্ট: save and retrieve।
* **ফিচারের অসুবিধা বা কমতি**: Storage growth (large history), query performance slow, incomplete logs, এবং sync issues (cloud/local)।
* **ফিচারের অসুবিধা দূর করার জন্য কি ব্যবস্থা গ্রহণ করা যায়**: Growth-এর জন্য archive old। Perf-এর জন্য indexing। Logs-এর জন্য detailed templates। Sync-এর জন্য existing sync engine।

### টাস্ক ১৯
* **ফিচারের নাম**: Smart Prompt Chaining (For Sensitive Tasks)
* **ফিচারের বিস্তারিত বর্ণনা**: এই ফিচারটি সেনসিটিভ টাস্ক (e.g., file access) এর জন্য প্রম্পট চেইনিং করবে: multi-step prompts (plan, review, execute) যাতে accuracy বাড়ে এবং errors কম হয়।
* **ফিচারটি কিভাবে ইমপ্লিমেন্ট করা হবে**: 
  - **লজিক**: PromptChainer ক্লাসে chain_prompts: list of prompts sequential call to LLM, refine outputs।
  - **লাইব্রেরী এবং টুলস**: langchain (chaining, if available; else manual loops), requests (LLM)।
  - **স্টেপ-বাই-স্টেপ ইমপ্লিমেন্টেশন**: 1. Chainer class create। 2. Sensitive flag in tasks। 3. Integrate with engine। 4. টেস্ট: chain for dummy sensitive task।
* **ফিচারের অসুবিধা বা কমতি**: Slower (multiple calls), higher cost (API usage), chain breaks (one step fail), এবং over-complex for simple tasks।
* **ফিচারের অসুবিধা দূর করার জন্য কি ব্যবস্থা গ্রহণ করা যায়**: Slow-এর জন্য parallel chains (if possible)। Cost-এর জন্য local model priority। Breaks-এর জন্য error recovery। Complex-এর জন্য conditional chaining।

### টাস্ক ২০
* **ফিচারের নাম**: Final Test: Check if System Adds TTS on "Add TTS" Command
* **ফিচারের বিস্তারিত বর্ণনা**: এই ফিচারটি ফাইনাল টেস্ট: ইউজার কমান্ড "TTS যোগ করো" দিলে সিস্টেম নিজে TTS endpoint যোগ করে কিনা চেক। End-to-end validation, report with metrics।
* **ফিচারটি কিভাবে ইমপ্লিমেন্ট করা হবে**: 
  - **লজিক**: Integration test script: simulate user input, run upgrade, verify new endpoint functional (e.g., TTS lib integration)।
  - **লাইব্রেরী এবং টুলস**: pytest, gtts/pyttsx3 (TTS), requests (endpoint test)।
  - **স্টেপ-বাই-স্টেপ ইমপ্লিমেন্টেশন**: 1. Test script write। 2. TTS code template। 3. Run full flow। 4. Analyze results for phase completion।
* **ফিচারের অসুবিধা বা কমতি**: Test flaky (env differences), incomplete verification (edge cases), time-intensive, এবং requires real deployment।
* **ফিচারের অসুবিধা দূর করার জন্য কি ব্যবস্থা গ্রহণ করা যায়**: Flaky-এর জন্য retries in tests। Verification-এর জন্য more asserts। Time-এর জন্য automated CI। Deployment-এর জন্য mock Render।
### টাস্ক ২১
* **ফিচারের নাম**: SelfLearningManager (Save Lessons from Errors)
* **ফিচারের বিস্তারিত বর্ণনা**: এই ফিচারটি একটা ম্যানেজার ক্লাস তৈরি করবে যা সিস্টেমের ভুল বা এরর থেকে লেসন এক্সট্র্যাক্ট করে লং-টার্ম মেমরিতে সেভ করবে। এটি অটোমেটিকভাবে ট্রিগার হবে যখন এরর হবে, লেসনগুলোকে ক্যাটাগরাইজ করবে (e.g., 'code_error', 'performance_issue'), এবং ফিউচার ডিসিশনে ব্যবহার করবে যাতে একই ভুল দ্বিতীয়বার না হয়।
* **ফিচারটি কিভাবে ইমপ্লিমেন্ট করা হবে**: 
  - **লজিক**: SelfLearningManager ক্লাসে learn_from_error মেথড যোগ, যা এরর কনটেক্সট অ্যানালাইজ করে (e.g., LLM prompt: "What lesson from this error?") লেসন জেনারেট করে মেমরিতে সেভ। Before action, check_lessons to avoid known issues।
  - **লাইব্রেরী এবং টুলস**: logging (error context), json (lesson format), TaskMemoryManager (storage integrate)। Groq/Ollama for analysis prompts।
  - **স্টেপ-বাই-স্টেপ ইমপ্লিমেন্টেশন**: 1. SelfLearningManager.py তৈরি। 2. Error hook from ErrorLogger। 3. Lesson generation logic। 4. টেস্ট: simulate error and check saved lesson।
* **ফিচারের অসুবিধা বা কমতি**: Lesson generation inaccurate (LLM hallucinations), storage overload from too many lessons, irrelevant lessons applied wrongly, এবং manual review absence।
* **ফিচারের অসুবিধা দূর করার জন্য কি ব্যবস্থা গ্রহণ করা যায়**: Accuracy-এর জন্য multi-LLM verification। Overload-এর জন্য priority scoring and pruning। Wrong application-এর জন্য confidence thresholds। Review-এর জন্য user feedback loop।

### টাস্ক ২২
* **ফিচারের নাম**: Long-Term Task State Management (Runs for Years)
* **ফিচারের বিস্তারিত বর্ণনা**: এই ফিচারটি লং-টার্ম টাস্কগুলোর স্টেট ম্যানেজ করবে, যেমন প্রোগ্রেস, ডিপেন্ডেন্সি, এবং রিজুম ক্যাপাবিলিটি, যাতে বছরের পর বছর চলতে পারে (e.g., ongoing learning projects)। এটি পিরিয়ডিক চেকপয়েন্ট সেভ করবে এবং রিস্টার্টে রিজুম করবে।
* **ফিচারটি কিভাবে ইমপ্লিমেন্ট করা হবে**: 
  - **লজিক**: TaskStateManager ক্লাসে save_state, load_state মেথড: state as JSON (progress, variables) in DB। Scheduler for periodic saves (e.g., every hour for long tasks)।
  - **লাইব্রেরী এবং টুলস**: sqlite3/json (state storage), schedule/APScheduler (periodic tasks), logging।
  - **স্টেপ-বাই-স্টেপ ইমপ্লিমেন্টেশন**: 1. TaskStateManager.py তৈরি। 2. DB schema for tasks (id, state, last_update)। 3. Integrate with core engine। 4. টেস্ট: long-running sim and resume।
* **ফিচারের অসুবিধা বা কমতি**: State corruption over time, large states slow loading, dependency changes break resume, এবং security (sensitive state data)।
* **ফিচারের অসুবিধা দূর করার জন্য কি ব্যবস্থা গ্রহণ করা যায়**: Corruption-এর জন্য versioning/checksums। Slow-এর জন্য compression (gzip)। Breaks-এর জন্য validation on load। Security-এর জন্য encryption (cryptography)।

### টাস্ক ২৩
* **ফিচারের নাম**: Hybrid Learning (Learn from Local + Groq)
* **ফিচারের বিস্তারিত বর্ণনা**: এই ফিচারটি হাইব্রিড লার্নিং ইমপ্লিমেন্ট করবে, যেখানে লোকাল 14B মডেল লোকাল ডাটা থেকে শিখবে এবং Groq 70B ক্লাউড-বেসড জ্ঞান থেকে, তারপর কম্বাইন করে লেসন তৈরি করবে। এটি অপটিমাইজড সুইচিং ব্যবহার করে খরচ এবং প্রাইভেসি ব্যালেন্স করবে।
* **ফিচারটি কিভাবে ইমপ্লিমেন্ট করা হবে**: 
  - **লজিক**: HybridLearner ক্লাসে learn_hybrid: local_query (Ollama), cloud_query (Groq), then merge_results (e.g., vote or average confidence)। Integrate with SelfLearningManager।
  - **লাইব্রেরী এবং টুলস**: ollama/requests (models), json (result merge), logging।
  - **স্টেপ-বাই-স্টেপ ইমপ্লিমেন্টেশন**: 1. HybridLearner.py তৈরি। 2. Query functions define। 3. Merge logic (e.g., similarity check)। 4. টেস্ট: sample learning query।
* **ফিচারের অসুবিধা বা কমতি**: Merge conflicts (contradictory info), higher latency (dual queries), cost increase (Groq usage), এবং privacy leak to cloud।
* **ফিচারের অসুবিধা দূর করার জন্য কি ব্যবস্থা গ্রহণ করা যায়**: Conflicts-এর জন্য priority rules (local first)। Latency-এর জন্য async parallel queries। Cost-এর জন্য threshold-based cloud use। Privacy-এর জন্য anonymize data before cloud。

### টাস্ক ২৪
* **ফিচারের নাম**: Auto Summarization (Compact Old Memories)
* **ফিচারের বিস্তারিত বর্ণনা**: এই ফিচারটি পুরনো মেমরিগুলোকে অটোমেটিক সামারাইজ করে কমপ্যাক্ট করবে, যাতে স্টোরেজ অপটিমাইজ হয় এবং রিট্রিভাল ফাস্ট হয়। এটি পিরিয়ডিক রান করবে (e.g., monthly), অরিজিনাল ডাটা আর্কাইভ করে সামারি সেভ করবে।
* **ফিচারটি কিভাবে ইমপ্লিমেন্ট করা হবে**: 
  - **লজিক**: Summarizer ক্লাসে summarize_memories: fetch old (e.g., >6 months), LLM prompt for summary, save summary and archive original।
  - **লাইব্রেরী এবং টুলস**: schedule (periodic), json/sqlite3 (archive), Groq for summarization prompts।
  - **স্টেপ-বাই-স্টেপ ইমপ্লিমেন্টেশন**: 1. Summarizer.py তৈরি। 2. Query old memories। 3. Prompt template। 4. টেস্ট: sample summaries।
* **ফিচারের অসুবিধা বা কমতি**: Information loss in summaries, summarization errors (LLM bias), compute intensive, এবং archive access slow।
* **ফিচারের অসুবিধা দূর করার জন্য কি ব্যবস্থা গ্রহণ করা যায়**: Loss-এর জন্য key points extraction। Errors-এর জন্য human review option। Intensive-এর জন্য batch processing। Access-এর জন্য indexed archives।

### টাস্ক ২৫
* **ফিচারের নাম**: Final Security Audit + Rule-Based Approval
* **ফিচারের বিস্তারিত বর্ণনা**: এই ফিচারটি সম্পূর্ণ সিস্টেমের সিকিউরিটি অডিট করবে এবং রুল-বেসড অ্যাপ্রুভাল সিস্টেম আপগ্রেড করবে, যেমন predefined rules for actions (e.g., no delete without backup)। অডিট রিপোর্ট জেনারেট করবে vulnerabilities সহ।
* **ফিচারটি কিভাবে ইমপ্লিমেন্ট করা হবে**: 
  - **লজিক**: SecurityAuditor ক্লাসে run_audit: scan code (static analysis), test vulnerabilities। ApprovalManager-এ rule engine (if-then rules)।
  - **লাইব্রেরী এবং টুলস**: bandit/pylint (security scan), yaml (rule files), logging/reportlab (reports)।
  - **স্টেপ-বাই-স্টেপ ইমপ্লিমেন্টেশন**: 1. Auditor class create। 2. Rule parser। 3. Integrate scans। 4. টেস্ট: mock vulnerabilities।
* **ফিচারের অসুবিধা বা কমতি**: False positives in scans, rules outdated, audit time-consuming, এবং zero-day vulnerabilities missed।
* **ফিচারের অসুবিধা দূর করার জন্য কি ব্যবস্থা গ্রহণ করা যায়**: Positives-এর জন্য custom suppressions। Outdated-এর জন্য auto-update rules। Time-এর জন্য scheduled audits। Missed-এর জন্য external tools integration।

### টাস্ক ২৬
* **ফিচারের নাম**: Plugin System Base Creation (For Future Use)
* **ফিচারের বিস্তারিত বর্ণনা**: এই ফিচারটি প্লাগিন সিস্টেমের বেস তৈরি করবে, যাতে ফিউচারে নতুন ফিচার (e.g., TTS, hacking tools) প্লাগিন হিসেবে যোগ করা যায়। এটি লোডার, রেজিস্ট্রি, এবং ইন্টারফেস ডিফাইন করবে।
* **ফিচারটি কিভাবে ইমপ্লিমেন্ট করা হবে**: 
  - **লজিক**: PluginManager ক্লাসে load_plugin (dynamic import), register (dict of plugins)। BasePlugin abstract class for interface。
  - **লাইব্রেরী এবং টুলস**: importlib (dynamic load), abc (abstract base), logging。
  - **স্টেপ-বাই-স্টেপ ইমপ্লিমেন্টেশন**: 1. PluginManager.py create। 2. Base class define। 3. Sample dummy plugin। 4. টেস্ট: load and call。
* **ফিচারের অসুবিধা বা কমতি**: Plugin conflicts, security risks (malicious plugins), loading overhead, এবং compatibility issues।
* **ফিচারের অসুবিধা দূর করার জন্য কি ব্যবস্থা গ্রহণ করা যায়**: Conflicts-এর জন্য dependency checker। Risks-এর জন্য sandboxing। Overhead-এর জন্য lazy loading। Compatibility-এর জন্য versioning。

### টাস্ক ২৭
* **ফিচারের নাম**: Voice Interface Preparation (TTS/STT)
* **ফিচারের বিস্তারিত বর্ণনা**: এই ফিচারটি ভয়েস ইন্টারফেসের প্রস্তুতি করবে: TTS (Text-to-Speech) এবং STT (Speech-to-Text) ইন্টিগ্রেশন, যাতে সিস্টেম ভয়েস কমান্ড নিতে এবং রেসপন্ড করতে পারে। প্রথমে বেসিক API wrappers তৈরি।
* **ফিচারটি কিভাবে ইমপ্লিমেন্ট করা হবে**: 
  - **লজিক**: VoiceInterface ক্লাসে tts (text to audio), stt (audio to text)। Integrate with core chat for voice mode।
  - **লাইব্রেরী এবং টুলস**: gtts/pyttsx3 (TTS), speech_recognition (STT), pydub (audio handling)।
  - **স্টেপ-বাই-স্টেপ ইমপ্লিমেন্টেশন**: 1. Interface class create। 2. Methods implement। 3. Audio file handling। 4. টেস্ট: text to speech and back。
* **ফিচারের অসুবিধা বা কমতি**: Accuracy low in noisy env, language limitations, privacy (audio data), এবং resource heavy (real-time)।
* **ফিচারের অসুবিধা দূর করার জন্য কি ব্যবস্থা গ্রহণ করা যায়**: Accuracy-এর জন্য noise reduction (libs)। Language-এর জন্য multi-lang support। Privacy-এর জন্য local processing। Heavy-এর জন্য async threads。

### টাস্ক ২৮
* **ফিচারের নাম**: Dashboard UI (Streamlit)
* **ফিচারের বিস্তারিত বর্ণনা**: এই ফিচারটি একটা ড্যাশবোর্ড UI তৈরি করবে Streamlit দিয়ে, যেখানে স্ট্যাটাস, মেমরি, আপগ্রেড ইতিহাস, এবং কন্ট্রোল (e.g., approve buttons) দেখা যাবে। এটি ইউজার-ফ্রেন্ডলি হবে।
* **ফিচারটি কিভাবে ইমপ্লিমেন্ট করা হবে**: 
  - **লজিক**: Streamlit app.py: pages for status, memory query, approvals। Data from API endpoints (e.g., /status)।
  - **লাইব্রেরী এবং টুলস**: streamlit (UI), pandas (data display), requests (backend calls)।
  - **স্টেপ-বাই-স্টেপ ইমপ্লিমেন্টেশন**: 1. App script create। 2. Components add (tables, buttons)। 3. Auth for security। 4. টেস্ট: local run।
* **ফিচারের অসুবিধা বা কমতি**: UI slow for large data, security (exposed endpoints), dependency on Streamlit, এবং mobile unfriendly।
* **ফিচারের অসুবিধা দূর করার জন্য কি ব্যবস্থা গ্রহণ করা যায়**: Slow-এর জন্য pagination। Security-এর জন্য HTTPS/auth। Dependency-এর জন্য alternatives (future)। Mobile-এর জন্য responsive design।

### টাস্ক ২৯
* **ফিচারের নাম**: Full System End-to-End Test (1 Month Long Task)
* **ফিচারের বিস্তারিত বর্ণনা**: এই ফিচারটি সম্পূর্ণ সিস্টেমের এন্ড-টু-এন্ড টেস্ট করবে, একটা ১ মাসের লং টাস্ক সিমুলেট করে (e.g., ongoing learning)। কভার: stability, memory, upgrades, etc. রিপোর্ট সহ।
* **ফিচারটি কিভাবে ইমপ্লিমেন্ট করা হবে**: 
  - **লজিক**: E2ETestSuite: long-running script with checkpoints, monitor metrics (uptime, errors)।
  - **লাইব্রেরী এবং টুলস**: pytest/selenium (if UI), time (simulation), logging/prometheus (metrics)।
  - **স্টেপ-বাই-স্টেপ ইমপ্লিমেন্টেশন**: 1. Test suite script। 2. Scenarios define। 3. Run in staging। 4. Analyze report।
* **ফিচারের অসুবিধা বা কমতি**: Time-consuming (real 1 month?), incomplete coverage, env differences, এবং resource drain।
* **ফিচারের অসুবিধা দূর করার জন্য কি ব্যবস্থা গ্রহণ করা যায়**: Time-এর জন্য accelerated sim। Coverage-এর জন্য automated generators। Env-এর জন্য docker。 Drain-এর জন্য cloud resources।

### টাস্ক ৩০
* **ফিচারের নাম**: v1.0 Release + Documentation
* **ফিচারের বিস্তারিত বর্ণনা**: এই ফিচারটি v1.0 রিলিজ করবে: final build, deployment, এবং ডকুমেন্টেশন (setup, usage, API docs)। GitHub release with changelog।
* **ফিচারটি কিভাবে ইমপ্লিমেন্ট করা হবে**: 
  - **লজিক**: Release script: bump version, generate docs (e.g., sphinx), push to Render/GitHub।
  - **লাইব্রেরী এবং টুলস**: sphinx/mkdocs (docs), gitpython (release), setuptools (packaging)।
  - **স্টেপ-বাই-স্টেপ ইমপ্লিমেন্টেশন**: 1. Docs folder setup। 2. Release automation script। 3. Changelog generate। 4. Deploy and test।
* **ফিচারের অসুবিধা বা কমতি**: Docs outdated quickly, release bugs, distribution issues, এবং user adoption low।
* **ফিচারের অসুবিধা দূর করার জন্য কি ব্যবস্থা গ্রহণ করা যায়**: Outdated-এর জন্য auto-gen docs। Bugs-এর জন্য beta testing। Distribution-এর জন্য PyPI/Docker। Adoption-এর জন্য tutorials।


