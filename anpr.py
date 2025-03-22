from fastapi import FastAPI, BackgroundTasks
import threading
import queue
import cv2
import time
import logging
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from paddleocr import PaddleOCR
import database
from config import (OUTPUT_DIR, RTSP_URL, YOLO_MODEL_PATH, CONF_THRESHOLD,
                    FRAME_QUEUE_SIZE, PROCESSED_QUEUE_SIZE, PLATE_PATTERN1, PLATE_PATTERN2, GATE_NO, LANE_NO, IN_OUT_TYPE)
from detector import detect_plates
from ocr_utils import extract_plate, read_plate_text

# Setup FastAPI
app = FastAPI()

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Global variables
frame_queue = queue.Queue(maxsize=FRAME_QUEUE_SIZE)
db_queue = queue.Queue()
stop_event = threading.Event()

# Load Models
def load_models():
    global vehicle_model, plate_model, tracker, ocr_reader
    vehicle_model = YOLO(model="yolo11n.pt").to('cuda')
    plate_model = YOLO(YOLO_MODEL_PATH).to('cuda')
    tracker = DeepSort(max_age=30)
    ocr_reader = PaddleOCR(lang="en", show_log=False, use_gpu=True)

def capture_frames():
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        logging.error("Unable to open video stream")
        stop_event.set()
        return
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(2)
            cap = cv2.VideoCapture(RTSP_URL)
            continue
        frame_queue.put(frame)
    cap.release()

def process_frames():
    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=1)
        except queue.Empty:
            continue
        results = vehicle_model(frame, conf=CONF_THRESHOLD, classes=[2, 5, 7])
        detections = [[box.xyxy[0].tolist(), float(box.conf[0]), 'vehicle'] for result in results for box in result.boxes]
        tracks = tracker.update_tracks(detections, frame=frame)
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            vehicle_roi = frame[y1:y2, x1:x2]
            plate_detections = detect_plates(vehicle_roi, plate_model, CONF_THRESHOLD)
            for px1, py1, px2, py2, pconf in plate_detections:
                px1, py1, px2, py2 = px1 + x1, py1 + y1, px2 + x1, py2 + y1
                plate_img = extract_plate(frame, (px1, py1, px2, py2))
                plate_text, ocr_score = read_plate_text(plate_img, ocr_reader)
                valid_plate = plate_text.upper() if PLATE_PATTERN1.fullmatch(plate_text.upper()) or PLATE_PATTERN2.fullmatch(plate_text.upper()) else "INVALID"
                timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                mean_score = ((0.60 * float(pconf)) + (0.40 * float(ocr_score))) / 2
                db_queue.put((timestamp_str, pconf, valid_plate, ocr_score, GATE_NO, LANE_NO, IN_OUT_TYPE, track_id, mean_score))

def start_anpr():
    database.create_table()
    load_models()
    global capture_thread, process_thread, db_thread
    capture_thread = threading.Thread(target=capture_frames, daemon=True)
    process_thread = threading.Thread(target=process_frames, daemon=True)
    db_thread = threading.Thread(target=database.db_writer_thread, args=(db_queue, stop_event), daemon=True)
    capture_thread.start()
    process_thread.start()
    db_thread.start()

@app.post("/start")
def start_anpr_endpoint(background_tasks: BackgroundTasks):
    stop_event.clear()
    background_tasks.add_task(start_anpr)
    return {"status": "ANPR system started"}

@app.post("/stop")
def stop_anpr():
    stop_event.set()
    return {"status": "ANPR system stopped"}

@app.get("/status")
def get_status():
    return {"status": "Running" if not stop_event.is_set() else "Stopped"}

# Dockerfile
DOCKERFILE = '''
FROM python:3.9-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
'''

# Save Dockerfile
with open("Dockerfile", "w") as f:
    f.write(DOCKERFILE)

# Render Deployment Instructions
RENDER_DEPLOYMENT = '''
# Steps to Deploy on Render:
1. Create a new **Web Service** on Render.
2. Connect your GitHub repository.
3. Set the **Build Command**: `pip install -r requirements.txt`
4. Set the **Start Command**: `uvicorn main:app --host 0.0.0.0 --port 8000`
5. Select a free instance and deploy!
'''
print(RENDER_DEPLOYMENT)
