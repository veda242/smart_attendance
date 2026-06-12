# 🎓 SmartAttend - AI Powered Smart Attendance System

An intelligent face recognition-based attendance management system built using **Python, Flask, OpenCV, MediaPipe, FaceNet, and PyTorch**. The system automatically detects and recognizes students in real-time through a webcam and marks attendance without manual intervention.

## 🚀 Features

* Real-time face detection using MediaPipe
* Face recognition using FaceNet embeddings
* Automatic attendance marking
* Live webcam monitoring dashboard
* Subject-wise attendance sessions
* Attendance records stored automatically
* Responsive web interface using Flask
* Fast and efficient student identification

---

## 🛠️ Tech Stack

### Backend

* Python
* Flask
* Flask-SocketIO

### Computer Vision & AI

* OpenCV
* MediaPipe
* FaceNet-PyTorch
* PyTorch
* NumPy

### Frontend

* HTML
* CSS
* JavaScript

---

## 📂 Project Structure

```bash
smart_attendance/
│
├── app.py
├── recognition.py
├── scheduler.py
├── collect_faces.py
├── augment_and_embed.py
├── requirements.txt
│
├── static/
├── templates/
│
├── train_images/      # User face images (not uploaded)
├── embeddings/        # Generated embeddings (not uploaded)
├── attendance/        # Attendance records
│
└── README.md
```

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/<your-username>/smart_attendance.git
cd smart_attendance
```

### Create Virtual Environment

```bash
python -m venv .venv
```

### Activate Environment

Windows:

```bash
.venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📸 Register Student Faces

Collect face samples:

```bash
python collect_faces.py
```

Generate embeddings:

```bash
python augment_and_embed.py
```

---

## ▶️ Run Application

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

---

## 🔄 Working Flow

1. Capture student face images.
2. Generate FaceNet embeddings.
3. Start attendance session.
4. Detect faces using MediaPipe.
5. Extract embeddings using FaceNet.
6. Match embeddings with enrolled students.
7. Automatically mark attendance.
8. Store attendance records.

---

## 📊 Key Highlights

* Real-time face recognition attendance system.
* Eliminates proxy attendance.
* Reduces manual effort and paperwork.
* Scalable for classrooms, labs, and training centers.
* Uses deep learning-based facial embeddings for robust identification.

---

## 🔒 Privacy Note

Face images, generated embeddings, and attendance records are excluded from this repository to protect user privacy and sensitive biometric data.

---



```
```
