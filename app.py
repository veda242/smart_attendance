"""
app.py — Flask web application.
Streams live video, marks attendance, exposes REST API for dashboard.
"""

import os
import sys
import csv
import datetime
import threading
import cv2

# Disable torch.dynamo before importing torch-based libraries
os.environ['TORCH_COMPILE'] = 'False'
os.environ['DISABLE_TELEMETRY'] = 'True'

from flask import Flask, Response, render_template, jsonify, request
from flask_socketio import SocketIO, emit

from recognition import AttendanceEngine

# ── Flask setup ───────────────────────────────────────────────────────────────
app     = Flask(__name__)
app.config["SECRET_KEY"] = "smart_attendance_2024"
socketio = SocketIO(app, cors_allowed_origins="*")

ATTENDANCE_DIR = "attendance"
os.makedirs(ATTENDANCE_DIR, exist_ok=True)

# ── Global state ──────────────────────────────────────────────────────────────
engine        = None
camera        = None
session_active = False
session_subject = ""
lock          = threading.Lock()


def get_attendance_file():
    today = datetime.date.today().isoformat()
    return os.path.join(ATTENDANCE_DIR, f"attendance_{today}.csv")


def save_to_csv(name: str, subject: str):
    filepath = get_attendance_file()
    file_exists = os.path.exists(filepath)
    with open(filepath, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Name", "Subject", "Date", "Time", "Status"])
        now = datetime.datetime.now()
        writer.writerow([name, subject, now.strftime("%Y-%m-%d"),
                         now.strftime("%H:%M:%S"), "Present"])
    print(f"[CSV] Marked {name} present for {subject}")


# ── Video streaming ───────────────────────────────────────────────────────────
def generate_frames():
    global engine, camera, session_active
    while True:
        if not session_active or camera is None:
            import time; time.sleep(0.1)
            continue

        ret, frame = camera.read()
        if not ret:
            break

        with lock:
            annotated, newly_marked = engine.process_frame(frame)

        # Notify clients via SocketIO for real-time dashboard update
        if newly_marked:
            for name in newly_marked:
                save_to_csv(name, session_subject)
            socketio.emit("attendance_update", {
                "newly_marked": newly_marked,
                "all_marked": list(engine.marked)
            })

        # Encode frame as JPEG
        _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/start", methods=["POST"])
def start_session():
    global engine, camera, session_active, session_subject
    data = request.json or {}
    session_subject = data.get("subject", "General")

    if session_active:
        return jsonify({"status": "already_running"})

    try:
        print("[START] ============================================")
        print("[START] Initiating session for subject:", session_subject)
        print("[START] Creating AttendanceEngine...")
        sys.stdout.flush()
        
        engine = AttendanceEngine()
        print("[START] AttendanceEngine created successfully")
        sys.stdout.flush()
        
        # Try multiple camera indices (0, 1, 2)
        camera_opened = False
        for camera_index in range(3):
            print(f"[START] Trying camera index {camera_index}...")
            sys.stdout.flush()
            camera = cv2.VideoCapture(camera_index)
            if camera.isOpened():
                print(f"[START] ✓ Camera {camera_index} opened successfully")
                camera_opened = True
                break
            else:
                print(f"[START] ✗ Camera {camera_index} failed to open")
                camera.release()
        
        sys.stdout.flush()
        
        if not camera_opened:
            error_msg = "No camera found. Try restarting the app."
            print(f"[START] Error: {error_msg}")
            sys.stdout.flush()
            return jsonify({"status": "error", "message": error_msg}), 500
        
        print("[START] Setting camera properties...")
        sys.stdout.flush()
        camera.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        session_active = True
        print(f"[START] ✓ Session started for subject: {session_subject}")
        print("[START] ============================================")
        sys.stdout.flush()
        return jsonify({"status": "started", "subject": session_subject})
    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"[START] ============ ERROR ============")
        print(f"[START] Error occurred: {error_msg}")
        traceback.print_exc()
        print("[START] ====================================")
        sys.stdout.flush()
        return jsonify({"status": "error", "message": error_msg}), 500


@app.route("/api/stop", methods=["POST"])
def stop_session():
    global camera, session_active
    session_active = False
    if camera:
        camera.release()
        camera = None
    return jsonify({"status": "stopped",
                    "marked": list(engine.marked) if engine else []})


@app.route("/api/status")
def get_status():
    if not engine or not session_active:
        return jsonify({"active": False, "marked": [], "timers": {}})
    status = engine.get_status()
    status["active"] = True
    status["subject"] = session_subject
    return jsonify(status)


@app.route("/api/attendance")
def get_attendance():
    """Return today's attendance CSV as JSON."""
    filepath = get_attendance_file()
    records = []
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            reader = csv.DictReader(f)
            records = list(reader)
    return jsonify({"date": datetime.date.today().isoformat(), "records": records})


@app.route("/api/attendance/all")
def get_all_attendance():
    """Return list of all attendance files."""
    files = sorted(os.listdir(ATTENDANCE_DIR), reverse=True)
    return jsonify({"files": files})


# ── SocketIO ──────────────────────────────────────────────────────────────────
@socketio.on("connect")
def on_connect():
    emit("connected", {"message": "Connected to Smart Attendance System"})


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)