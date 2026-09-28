"""
mineRakshak — Python backend (FastAPI)

Serves the dashboard UI and provides:
  - REST API for trip control and safety-state
  - A WebSocket that pushes live safety-state updates to the dashboard
  - Real, live MJPEG video streaming from your external camera
      GET /camera/feed           -> normal live feed
      GET /camera/feed/thermal   -> same live feed, false-colour "thermal" look

Run:
    pip install -r requirements.txt
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Then open:  http://127.0.0.1:8000
Swagger UI: http://127.0.0.1:8000/docs   (use POST /api/dashboard to
            manually push safety conditions, just like the old
            "Swagger is the source of truth" workflow)
"""

import asyncio
import json
import os
import random
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent

# ----------------------------------------------------------------------------
# Camera source
#
# USB webcams don't always land on the same /dev/videoN index — it can shift
# depending on the USB port, boot order, or whether other cameras are also
# plugged in. The vendor:product ID, however, never changes for a given
# physical camera, so we match on that instead of a hardcoded index.
#
#   Bus 001 Device 010: ID 1908:2310 GEMBIRD USB2.0 PC CAMERA
#
# Set the MINERAKSHAK_CAMERA env var to override this entirely, e.g.
#   MINERAKSHAK_CAMERA=1                  (force a specific index)
#   MINERAKSHAK_CAMERA=rtsp://...         (an IP camera instead)
# ----------------------------------------------------------------------------
GEMBIRD_USB_VENDOR_ID = "1908"
GEMBIRD_USB_PRODUCT_ID = "2310"


def find_video_index_by_usb_id(vendor_id: str, product_id: str) -> Optional[int]:
    """Scan /sys/class/video4linux/videoN devices and return the index whose
    USB vendor:product ID matches. Returns None if not found (e.g. on
    non-Linux systems, or the camera isn't plugged in)."""
    v4l_dir = Path("/sys/class/video4linux")
    if not v4l_dir.exists():
        return None
    for entry in sorted(v4l_dir.iterdir(), key=lambda p: p.name):
        if not entry.name.startswith("video"):
            continue
        try:
            node = (entry / "device").resolve()
        except OSError:
            continue
        # Walk up from the video device node to the USB device directory
        # that actually carries idVendor/idProduct files.
        for _ in range(6):
            id_vendor_file = node / "idVendor"
            id_product_file = node / "idProduct"
            if id_vendor_file.exists() and id_product_file.exists():
                try:
                    found_vendor = id_vendor_file.read_text().strip().lower()
                    found_product = id_product_file.read_text().strip().lower()
                except OSError:
                    break
                if found_vendor == vendor_id.lower() and found_product == product_id.lower():
                    try:
                        return int(entry.name.replace("video", ""))
                    except ValueError:
                        return None
                break
            if node.parent == node:
                break
            node = node.parent
    return None


_env_source = os.environ.get("MINERAKSHAK_CAMERA")
if _env_source:
    try:
        DEFAULT_CAMERA_SOURCE = int(_env_source)
    except ValueError:
        DEFAULT_CAMERA_SOURCE = _env_source  # RTSP / HTTP URL / device path
else:
    _detected = find_video_index_by_usb_id(GEMBIRD_USB_VENDOR_ID, GEMBIRD_USB_PRODUCT_ID)
    DEFAULT_CAMERA_SOURCE = _detected if _detected is not None else 0


class CameraStream:
    """Grabs frames from the external camera in a background thread so the
    MJPEG endpoints always serve the freshest frame instead of blocking on
    cv2.read()."""

    def __init__(self, source):
        self.preferred_source = source
        self.source = source
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame = None
        self.lock = threading.Lock()
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.last_error: Optional[str] = None

    def start(self):
        if self.running:
            return
        candidates = [self.preferred_source]
        if isinstance(self.preferred_source, int):
            # auto-detect: an external webcam often isn't at index 0 if a
            # built-in camera is also present, so scan a few indices.
            candidates += [i for i in range(0, 5) if i != self.preferred_source]

        for src in candidates:
            cap = cv2.VideoCapture(src)
            if cap.isOpened():
                self.cap = cap
                self.source = src
                self.running = True
                self.last_error = None
                self.thread = threading.Thread(target=self._capture_loop, daemon=True)
                self.thread.start()
                return
            cap.release()

        self.last_error = f"No camera found (tried {candidates}). Set MINERAKSHAK_CAMERA to the right index or URL."

    def _capture_loop(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.4)
                continue
            with self.lock:
                self.frame = frame
            time.sleep(0.01)

    def get_frame(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()


camera = CameraStream(DEFAULT_CAMERA_SOURCE)


def _placeholder_frame(text: str):
    frame = np.zeros((480, 640, 3), dtype="uint8")
    frame[:] = (26, 20, 15)
    cv2.putText(frame, "mineRakshak", (150, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (239, 143, 47), 2)
    for i, line in enumerate(text[i:i + 46] for i in range(0, len(text), 46)):
        cv2.putText(frame, line, (40, 270 + i * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 200, 220), 1)
    return frame


# Some cameras (like this Gembird one) report frames upside down / sideways
# depending on how they're physically mounted. Rotate them right way up here.
# Override with MINERAKSHAK_CAMERA_ROTATE = 0, 90, 180, or 270.
_ROTATE_MAP = {
    0: None,
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}
try:
    CAMERA_ROTATE_DEGREES = int(os.environ.get("MINERAKSHAK_CAMERA_ROTATE", "180"))
except ValueError:
    CAMERA_ROTATE_DEGREES = 180
_ROTATE_CODE = _ROTATE_MAP.get(CAMERA_ROTATE_DEGREES, cv2.ROTATE_180)


# ----------------------------------------------------------------------------
# YOLO person / vehicle detection
#
# Uses Ultralytics YOLOv8n (nano) — small enough to run on CPU. It's already
# trained on COCO, which covers "person" plus the vehicle classes relevant
# here (car, motorcycle, bus, truck), so no custom training is needed.
#
# The model file (yolov8n.pt, ~6 MB) downloads automatically the first time
# it's used, so the machine needs internet access once.
# ----------------------------------------------------------------------------
YOLO_MODEL_PATH = os.environ.get("MINERAKSHAK_YOLO_MODEL", "yolov8n.pt")
YOLO_CONF_THRESHOLD = float(os.environ.get("MINERAKSHAK_YOLO_CONF", "0.4"))
YOLO_INTERVAL_SEC = float(os.environ.get("MINERAKSHAK_YOLO_INTERVAL", "0.3"))

# COCO class ids: 0=person, 2=car, 3=motorcycle, 5=bus, 7=truck
YOLO_TARGET_CLASS_IDS = [0, 2, 3, 5, 7]
YOLO_VEHICLE_LABELS = {"car", "motorcycle", "bus", "truck"}

_yolo_model = None
_yolo_load_lock = threading.Lock()


def _get_yolo_model():
    """Load the model on first use (not at import time) so the server still
    starts even before ultralytics/the weights are ready."""
    global _yolo_model
    if _yolo_model is None:
        with _yolo_load_lock:
            if _yolo_model is None:
                from ultralytics import YOLO
                _yolo_model = YOLO(YOLO_MODEL_PATH)
    return _yolo_model


class Detection(BaseModel):
    label: str
    is_vehicle: bool
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int


detections_lock = threading.Lock()
latest_detections: list[Detection] = []
detection_error: Optional[str] = None


def _detection_loop():
    global latest_detections, detection_error
    camera.start()
    while True:
        time.sleep(YOLO_INTERVAL_SEC)
        frame = camera.get_frame()
        if frame is None:
            continue
        try:
            if _ROTATE_CODE is not None:
                frame = cv2.rotate(frame, _ROTATE_CODE)
            model = _get_yolo_model()
            results = model(frame, verbose=False, conf=YOLO_CONF_THRESHOLD, classes=YOLO_TARGET_CLASS_IDS)
            found = []
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                    label = model.names.get(int(box.cls[0]), "object")
                    found.append(Detection(
                        label=label,
                        is_vehicle=label in YOLO_VEHICLE_LABELS,
                        confidence=float(box.conf[0]),
                        x1=x1, y1=y1, x2=x2, y2=y2,
                    ))
            with detections_lock:
                latest_detections = found
                detection_error = None
        except Exception as exc:
            with detections_lock:
                detection_error = str(exc)


def _draw_detections(frame):
    with detections_lock:
        dets = list(latest_detections)
    for d in dets:
        color = (0, 165, 255) if d.is_vehicle else (60, 220, 60)  # BGR: orange for vehicles, green for people
        cv2.rectangle(frame, (d.x1, d.y1), (d.x2, d.y2), color, 2)
        text = f"{d.label} {d.confidence:.0%}"
        text_y = max(16, d.y1 - 8)
        cv2.putText(frame, text, (d.x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame


def _mjpeg_generator(mode: str = "normal"):
    camera.start()
    while True:
        frame = camera.get_frame()
        if frame is None:
            frame = _placeholder_frame(camera.last_error or "Connecting to external camera...")
        else:
            if _ROTATE_CODE is not None:
                frame = cv2.rotate(frame, _ROTATE_CODE)
            if mode == "thermal":
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = cv2.applyColorMap(255 - gray, cv2.COLORMAP_JET)
            frame = _draw_detections(frame)

        ok, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            continue
        chunk = jpeg.tobytes()
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + chunk + b"\r\n")
        time.sleep(0.04)  # ~25 fps


# ----------------------------------------------------------------------------
# Safety / dashboard state
# ----------------------------------------------------------------------------
class DashboardState(BaseModel):
    alert: str = "SYSTEM NORMAL"
    severity: str = "NORMAL"
    recommended_action: str = "CONTINUE"
    visibility_m: float = 500
    heavy_vehicle_detected: bool = False
    heavy_vehicle_distance_m: float = 0
    speed_kmh: float = 28


class TripStartRequest(BaseModel):
    vehicle_id: str = "TRUCK-01"


state = DashboardState()
state_lock = threading.Lock()
current_trip_id: Optional[str] = None
connected_sockets: list[WebSocket] = []


def get_state() -> dict:
    with state_lock:
        return state.dict()


def set_state(patch: dict):
    with state_lock:
        for k, v in patch.items():
            if hasattr(state, k) and v is not None:
                setattr(state, k, v)
        return state.dict()


async def broadcast_state():
    """Push the current state to every connected dashboard every second."""
    while True:
        await asyncio.sleep(1.0)
        payload = json.dumps({"type": "dashboard_update", "data": get_state()})
        stale = []
        for ws in connected_sockets:
            try:
                await ws.send_text(payload)
            except Exception:
                stale.append(ws)
        for ws in stale:
            connected_sockets.remove(ws)


def simulate_safety_loop():
    """Simple built-in simulator so the dashboard has something to react to
    even before you wire up real sensors. Overriding via POST /api/dashboard
    (Swagger UI) always takes priority — this loop only fills in when no
    trip is active or nothing else has been set recently."""
    scenarios = [
        dict(alert="SYSTEM NORMAL", severity="NORMAL", recommended_action="CONTINUE",
             heavy_vehicle_detected=False, heavy_vehicle_distance_m=0),
        dict(alert="HEAVY VEHICLE AHEAD", severity="HIGH", recommended_action="REDUCE SPEED",
             heavy_vehicle_detected=True, heavy_vehicle_distance_m=140),
        dict(alert="LOW VISIBILITY", severity="HIGH", recommended_action="SLOW DOWN",
             visibility_m=60),
    ]
    while True:
        time.sleep(random.uniform(12, 22))
        if current_trip_id is None:
            continue
        scenario = random.choice(scenarios)
        set_state(scenario)
        if scenario["alert"] == "SYSTEM NORMAL":
            set_state(dict(visibility_m=500, heavy_vehicle_detected=False))


# ----------------------------------------------------------------------------
# FastAPI app
# ----------------------------------------------------------------------------
app = FastAPI(title="mineRakshak Backend")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.on_event("startup")
async def on_startup():
    threading.Thread(target=simulate_safety_loop, daemon=True).start()
    threading.Thread(target=_detection_loop, daemon=True).start()
    asyncio.create_task(broadcast_state())


@app.get("/", response_class=HTMLResponse)
def dashboard_page():
    html_path = BASE_DIR / "templates" / "dashboard.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/dashboard")
def api_get_dashboard():
    return get_state()


@app.post("/api/dashboard")
def api_set_dashboard(patch: DashboardState):
    """Manually push a safety condition — this is the 'Swagger control' the
    dashboard trusts as ground truth."""
    return set_state(patch.dict())


@app.post("/api/trip/start")
def api_trip_start(req: TripStartRequest):
    global current_trip_id
    current_trip_id = f"TRIP-{int(time.time())}"
    set_state(dict(alert="SYSTEM NORMAL", severity="NORMAL", recommended_action="CONTINUE",
                    heavy_vehicle_detected=False, heavy_vehicle_distance_m=0, visibility_m=500))
    return {"trip_id": current_trip_id, "vehicle_id": req.vehicle_id}


@app.post("/api/trip/end")
def api_trip_end():
    global current_trip_id
    current_trip_id = None
    set_state(dict(alert="SYSTEM NORMAL", severity="NORMAL", recommended_action="CONTINUE",
                    heavy_vehicle_detected=False, heavy_vehicle_distance_m=0))
    return {"status": "ended"}


@app.websocket("/ws/dashboard")
async def ws_dashboard(websocket: WebSocket):
    await websocket.accept()
    connected_sockets.append(websocket)
    try:
        await websocket.send_text(json.dumps({"type": "dashboard_update", "data": get_state()}))
        while True:
            await websocket.receive_text()  # keep-alive / ignore inbound
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in connected_sockets:
            connected_sockets.remove(websocket)


@app.get("/camera/feed")
def camera_feed():
    return StreamingResponse(_mjpeg_generator("normal"),
                              media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/camera/feed/thermal")
def camera_feed_thermal():
    return StreamingResponse(_mjpeg_generator("thermal"),
                              media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/detections")
def get_detections():
    with detections_lock:
        dets = [d.dict() for d in latest_detections]
        err = detection_error
    return {
        "detections": dets,
        "person_count": sum(1 for d in dets if not d["is_vehicle"]),
        "vehicle_count": sum(1 for d in dets if d["is_vehicle"]),
        "error": err,
    }


@app.get("/camera/status")
def camera_status():
    camera.start()
    return {
        "running": camera.running,
        "source": camera.source,
        "error": camera.last_error,
        "rotation_degrees": CAMERA_ROTATE_DEGREES,
        "gembird_detected_via_usb_id": find_video_index_by_usb_id(
            GEMBIRD_USB_VENDOR_ID, GEMBIRD_USB_PRODUCT_ID
        ),
    }