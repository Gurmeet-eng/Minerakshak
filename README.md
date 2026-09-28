
## 1. Install

```bash
cd minerakshak
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```


## 2. Run

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Then open **http://127.0.0.1:8000** — this is the dashboard.

