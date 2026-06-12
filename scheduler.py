"""
scheduler.py — Auto-start/stop attendance sessions based on class timetable.
Edit TIMETABLE below to match your actual class schedule.
Run: python scheduler.py
"""

import schedule
import time
import requests
import datetime

BASE_URL = "http://localhost:5000"

# ── TIMETABLE ─────────────────────────────────────────────────────────────────
# Format: { "HH:MM": {"action": "start"/"stop", "subject": "..."} }
# All times in 24-hour format.

TIMETABLE = {
    # Monday–Friday
    "09:00": {"action": "start", "subject": "Data Structures"},
    "09:50": {"action": "stop",  "subject": ""},
    "10:00": {"action": "start", "subject": "Operating Systems"},
    "10:50": {"action": "stop",  "subject": ""},
    "11:00": {"action": "start", "subject": "Computer Networks"},
    "11:50": {"action": "stop",  "subject": ""},
    "14:00": {"action": "start", "subject": "DBMS"},
    "14:50": {"action": "stop",  "subject": ""},
    "15:00": {"action": "start", "subject": "Machine Learning"},
    "15:50": {"action": "stop",  "subject": ""},
}

# Days on which classes run (0=Mon, 1=Tue, ..., 6=Sun)
ACTIVE_DAYS = {0, 1, 2, 3, 4}   # Monday–Friday


# ── Helpers ───────────────────────────────────────────────────────────────────
def is_active_day():
    return datetime.datetime.today().weekday() in ACTIVE_DAYS


def trigger(action: str, subject: str = ""):
    if not is_active_day():
        print(f"[SCHEDULER] Skipping — not a class day.")
        return
    try:
        if action == "start":
            r = requests.post(f"{BASE_URL}/api/start", json={"subject": subject})
            print(f"[SCHEDULER] Started '{subject}' → {r.json()}")
        elif action == "stop":
            r = requests.post(f"{BASE_URL}/api/stop")
            print(f"[SCHEDULER] Stopped session → {r.json()}")
    except requests.ConnectionError:
        print("[SCHEDULER] ERROR: Flask app not running. Start app.py first.")


# ── Register all schedule entries ─────────────────────────────────────────────
for clock_time, info in TIMETABLE.items():
    action  = info["action"]
    subject = info.get("subject", "")
    # Bind defaults explicitly to avoid closure bug
    schedule.every().day.at(clock_time).do(
        lambda a=action, s=subject: trigger(a, s)
    )
    print(f"[SCHEDULER] Registered: {clock_time} → {action} '{subject}'")

print("\n[SCHEDULER] Running. Press Ctrl+C to stop.\n")

while True:
    schedule.run_pending()
    time.sleep(10)