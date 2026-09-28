# mineRakshak — Python backend + live camera

This is the original dashboard converted to a Python (FastAPI) web app. It now
also shows a **real, live video feed from your external camera** — not the
simulated gradient panel from before — in two places:

1. A small **"Live Camera" panel** docked in the bottom-right of the map,
   always visible.
2. The existing full-screen camera view (opened from the "Normal Camera" /
   "Thermal Camera" buttons) — "Thermal" now applies a false-colour heat-map
   to the same live feed, so it's a real thermal-style rendering, not a
   canned animation.

## 1. Install

```bash
cd minerakshak
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Point it at your external camera

By default the app tries index `0` and auto-scans `0-4` for the first camera
that opens — this usually finds an external USB webcam automatically. If it
picks the wrong one, set it explicitly before starting:

```bash
# macOS/Linux
export MINERAKSHAK_CAMERA=1        # try 1, 2, 3... for your external cam
# Windows (PowerShell)
$env:MINERAKSHAK_CAMERA=1
```

You can also point it at a network/IP camera by setting the same variable to
an RTSP/HTTP URL, e.g. `MINERAKSHAK_CAMERA=rtsp://192.168.1.20:554/stream1`.

Check what the backend actually connected to at any time:
`GET http://127.0.0.1:8000/camera/status`

## 3. Run

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Then open **http://127.0.0.1:8000** — this is the dashboard, served directly
by the Python backend (no separate HTML file to open).

Swagger UI (to manually push safety conditions, exactly like the old
"backend is the source of truth" workflow) is at **http://127.0.0.1:8000/docs**
— use `POST /api/dashboard` there to set `alert`, `severity`,
`heavy_vehicle_detected`, `visibility_m`, etc.

## What changed vs. the original HTML file

- All dashboard logic now lives behind a Python FastAPI server (`main.py`):
  trip start/end, the safety-state API, and a WebSocket that pushes live
  updates — this replaces the pure front-end simulation.
- `templates/dashboard.html` is the same UI, but the fake camera gradient was
  replaced by an `<img>` tag that streams real MJPEG video from
  `/camera/feed` (and `/camera/feed/thermal`) using OpenCV.
- A background thread grabs frames continuously so the video stays smooth
  even while multiple browser tabs are viewing it.
- A simple built-in simulator randomly raises "heavy vehicle" / "low
  visibility" alerts while a trip is active, purely so there's something to
  see out of the box — override it any time via Swagger.

## Notes / limits

- Only one process should hold the camera at a time; if another app has it
  open, `/camera/status` will report the error.
- MJPEG streaming works in every browser without extra plugins, but if you
  later want WebRTC-grade low latency for multiple remote viewers, that's a
  bigger follow-up (aiortc) — happy to help with that next if you need it.
