# config.py — সব env var এখানে এক জায়গায় রাখা হলো
import os

# Render & API
RENDER_URL = os.getenv("RENDER_URL", "https://the-mask-automation-core.onrender.com")
STATUS_AUTH_KEY = os.getenv("STATUS_AUTH_KEY", "secret_key_123")  # পরিবর্তন করতে পারো

# Approval & Timeout
APPROVAL_TIMEOUT = int(os.getenv("APPROVAL_TIMEOUT", "30"))  # সেকেন্ড

# Sync & Interval
SYNC_INTERVAL = int(os.getenv("SYNC_INTERVAL", "5"))  # মিনিট

# Plugins
PLUGIN_DIR = os.getenv("PLUGIN_DIR", "plugins/")

# TTS & Voice
TTS_TEST_URL = os.getenv("TTS_TEST_URL", "http://localhost:5000/tts")

# Dashboard
DASHBOARD_AUTH_PASSWORD = os.getenv("DASHBOARD_AUTH_PASSWORD", "secure_pass_2025")

# Task State
STATE_SAVE_INTERVAL = int(os.getenv("STATE_SAVE_INTERVAL", "3600"))  # সেকেন্ড (১ ঘণ্টা)

# Learning & Prompts
LEARNING_PROMPT = os.getenv("LEARNING_PROMPT", "Analyze this error: {error}. What lesson should the system learn? Categorize as code_error/performance_issue/security/etc.")

# Sandbox & Execution
SANDBOX_DIR = os.getenv("SANDBOX_DIR", "project_folder/")

# Backup & Restore
BACKUP_DIR = os.getenv("BACKUP_DIR", "backups/")