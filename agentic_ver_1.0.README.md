# telecom-agentic-demo — Detailed README

A minimal **agentic workflow** for telecom alarm triage and safe auto-remediation. It ships with:

* A Flask web app (“orchestrator”)
* Four concrete tools: `search_runbook`, `ask_kpi`, `run_remote`, `open_ticket/update_ticket`
* Dummy data for a realistic demo (alarms, KPIs, syslog) and two runbooks
* Optional hook to **Ollama** (defaults to a deterministic mock LLM for repeatable demos)

---

## 0) Prerequisites

* Python 3.10+
* (Optional) Ollama if you want a real LLM
* Git, Bash, and a browser

> Works on Linux/macOS/WSL. For Windows native, use PowerShell equivalents.

---

## 1) Quick Start (VS Code)

1. Open the repo in VS Code. Install the Python extension if prompted.
2. Bootstrap & run:

```bash
chmod +x run.sh
./run.sh
```

3. Visit [http://localhost:5057](http://localhost:5057)
4. Home → **Ingest →** for `DLK123`
5. Incident page → **Triage ▶** (mock LLM produces a deterministic JSON plan)
6. **Run: Load-shed** (simulated safe action) → ticket opens/updates
7. **Approve Restart** → **Run: Graceful Restart** (HITL flow)

### Use a real LLM via Ollama

```bash
ollama pull llama3.1:8b
export MOCK_LLM=0
export OLLAMA_MODEL=llama3.1:8b
./run.sh
```

---

## 2) What this demo teaches

* 🔁 Agentic loop: Ingest → Triage (RAG + policy) → Human approval → Action → Ticket
* 📚 RAG idea: tiny runbook search over `runbooks/*.md`
* 📈 KPI querying: moving-window average from CSV
* 🧪 Safety: only pre-approved actions; restart requires approval
* 👩‍💻 HITL: Approve/Run buttons

---

## 3) Architecture

```
Browser ──▶ Flask Orchestrator (app/orchestrator.py)
             │
             ├─ Tools
             │   ├─ Runbook Search  (app/tools/runbook_tool.py)
             │   ├─ KPI Query       (app/tools/kpi_tool.py)
             │   ├─ Remote Action   (app/tools/action_tool.py)
             │   └─ Ticketing       (app/tools/ticket_tool.py)
             │
             ├─ Data (app/data)
             │   ├─ alarms.csv
             │   ├─ kpis.csv
             │   ├─ syslog_DLK123.log
             │   └─ runbooks/*.md
             │
             └─ LLM (mock or Ollama)
```

* **orchestrator.py**: wires routes, loads data, invokes tools, and calls the LLM (mock by default).
* **Mock LLM** returns deterministic JSON for consistent demos.
* With `MOCK_LLM=0`, the app calls Ollama’s `/api/chat`.

---

## 4) Data Model (in-memory)

* `INCIDENTS[site_id] = {site_id, alarms, syslog, status, triage, actions, ticket}`
* `APPROVALS[(site_id, step)] = True/False`
* Tickets live in `ticket_tool.py` (`_TICKETS`).
  To avoid circular JSON, the incident keeps a **ticket summary** (no `details` field) when rendering.

---

## 5) Code Walkthrough (key files + snippets)

### 5.1 `app/__init__.py` (Flask factory)

```python
from flask import Flask
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RUNBOOK_DIR = DATA_DIR / "runbooks"
DB_PATH = BASE_DIR / "agent.db"

def create_app():
    app = Flask(__name__, template_folder=str(BASE_DIR / "templates"), static_folder=str(BASE_DIR / "static"))
    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret")
    app.config["DB_PATH"] = str(DB_PATH)
    app.config["DATA_DIR"] = str(DATA_DIR)
    app.config["RUNBOOK_DIR"] = str(RUNBOOK_DIR)
    return app
```

### 5.2 `app/tools/runbook_tool.py` (tiny runbook search)

```python
from pathlib import Path
import re

def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""

def search_runbook(query: str, runbook_dir: str) -> dict:
    q = query.lower()
    rb_dir = Path(runbook_dir)

    best = None
    best_score = -1
    best_snip = ""

    for md in rb_dir.glob("*.md"):
        txt = _read_text(md).lower()
        terms = re.findall(r"\\w+", q)
        score = sum(txt.count(t) for t in terms)
        if score > best_score:
            best = md
            best_score = score
            lines = _read_text(md).splitlines()
            hit = next((i for i, line in enumerate(lines) if terms and terms[0] in line.lower()), 0)
            best_snip = "\\n".join(lines[max(0, hit-2): hit+3])

    return {"query": query, "runbook": str(best.name) if best else None, "score": best_score, "snippet": best_snip}
```

### 5.3 `app/tools/kpi_tool.py` (moving-window KPI)

```python
import csv
from pathlib import Path
from datetime import datetime, timedelta

def _parse_iso(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00").replace("+00:00", "+00:00"))

def ask_kpi(metric: str, window: str, site_id: str, kpi_csv_path: str) -> dict:
    win_minutes = 30
    if window.endswith("m"):
        win_minutes = int(window[:-1])
    elif window.endswith("h"):
        win_minutes = int(window[:-1]) * 60

    rows = []
    kpi_path = Path(kpi_csv_path)
    with kpi_path.open() as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("site_id") == site_id:
                rows.append(row)

    if not rows:
        return {"metric": metric, "value": None, "count": 0}

    latest_ts = max(_parse_iso(r["ts"]) for r in rows)
    start_ts = latest_ts - timedelta(minutes=win_minutes)
    win_rows = [r for r in rows if start_ts <= _parse_iso(r["ts"]) <= latest_ts]

    vals = []
    for r in win_rows:
        try:
            vals.append(float(r.get(metric, 0.0)))
        except Exception:
            pass

    avg = sum(vals) / len(vals) if vals else None
    return {"metric": metric, "site_id": site_id, "window": window, "latest_ts": latest_ts.isoformat(), "samples": len(vals), "avg": avg}
```

### 5.4 `app/tools/action_tool.py` (safe simulated actions)

```python
from datetime import datetime

SAFE_ACTIONS = {
    "load_shed_on": "Applied load-shedding profile (reduce NR carrier power by 20%).",
    "radio_restart_graceful": "Graceful radio restart triggered.",
}

def run_remote(target: str, action: str, **kwargs) -> dict:
    now = datetime.utcnow().isoformat() + "Z"
    if action not in SAFE_ACTIONS:
        return {"ok": False, "target": target, "action": action, "ts": now, "result": f"Action '{action}' is not allowed in demo."}
    return {"ok": True, "target": target, "action": action, "ts": now, "result": SAFE_ACTIONS[action]}
```

### 5.5 `app/tools/ticket_tool.py` (in-memory tickets)

```python
from datetime import datetime

_TICKETS = {}

def open_ticket(title: str, severity: str, site_id: str, details: dict) -> dict:
    tid = f"TCK-{len(_TICKETS) + 1:04d}"
    _TICKETS[tid] = {
        "id": tid, "title": title, "severity": severity, "site_id": site_id,
        "details": details, "created_at": datetime.utcnow().isoformat() + "Z",
        "updates": [], "status": "OPEN",
    }
    return _TICKETS[tid]

def update_ticket(ticket_id: str, note: str, status: str | None = None) -> dict:
    t = _TICKETS.get(ticket_id)
    if not t:
        return {"ok": False, "error": "ticket not found"}
    t["updates"].append({"ts": datetime.utcnow().isoformat() + "Z", "note": note})
    if status:
        t["status"] = status
    return {"ok": True, "ticket": t}
```

### 5.6 `app/orchestrator.py` (routes & LLM wiring; anti-circular fix included)

**Env + mock/real LLM toggle**

```python
MOCK_LLM = os.getenv("MOCK_LLM", "1") == "1"
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
```

**LLM call**

```python
def call_llm(system_prompt: str, user_prompt: str) -> str:
    if MOCK_LLM:
        rb = search_runbook(user_prompt, app.config["RUNBOOK_DIR"])
        plan = {
            "cause": "Probable rectifier/battery issue with link flaps",
            "confidence": 0.72,
            "steps": [
                "Enable load_shed_on if battery < 30%",
                "Graceful radio restart when battery > 20%",
                "Open a ticket for onsite rectifier check if alarms persist",
            ],
            "runbook": rb,
        }
        return json.dumps(plan)

    url = f"{OLLAMA_HOST}/api/chat"
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
    }
    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data.get("message", {}).get("content", "{}")
```

**Triage route**

```python
@app.route("/triage", methods=["POST"])
def triage():
    site_id = request.form.get("site_id")
    inc = INCIDENTS.get(site_id)
    rb  = search_runbook("power fail link flap rectifier battery", app.config["RUNBOOK_DIR"])
    kpi = ask_kpi("rrc_drop_rate", "30m", site_id, str(Path(app.config["DATA_DIR"]) / "kpis.csv"))

    system_prompt = (
        "You are a telecom NOC triage assistant. Use safety-first planning. "
        "Return JSON with keys: cause, confidence [0-1], steps[], runbook."
    )
    user_prompt = (
        f"Site {site_id} has alarms: {inc['alarms']} and syslog hints. "
        f"KPI window avg: {kpi}. Prefer least-risk actions."
    )
    plan_raw = call_llm(system_prompt, user_prompt)

    try:
        plan = json.loads(plan_raw)
    except Exception:
        plan = {"cause": "Unknown", "confidence": 0.4, "steps": ["Open ticket"], "runbook": rb}

    inc["triage"] = {"plan": plan, "kpi": kpi, "rb": rb}
    inc["status"] = "TRIAGED"
    return redirect(url_for("incident", site_id=site_id))
```

**Action route (fixes circular reference)**

```python
@app.route("/action", methods=["POST"])
def action():
    site_id = request.form.get("site_id")
    step = request.form.get("step")  # allowed: load_shed_on, radio_restart_graceful
    inc = INCIDENTS.get(site_id)
    if not inc:
        return redirect(url_for("index"))

    if step == "radio_restart_graceful" and not APPROVALS.get((site_id, step)):
        return redirect(url_for("incident", site_id=site_id))

    res = run_remote(target=f"{site_id}.enodeb", action=step)

    # Open ticket on first action — store only a summary in ticket.details and
    # keep only a ticket summary (without 'details') in the incident to avoid cycles.
    if not inc.get("ticket"):
        safe_details = {
            "site_id": site_id,
            "status": inc.get("status"),
            "created_at": inc.get("created_at"),
            "alarms": inc.get("alarms"),
        }
        t = open_ticket(title=f"{site_id} triage", severity="S2", site_id=site_id, details=safe_details)
        inc["ticket"] = {k: v for k, v in t.items() if k != "details"}

    update_ticket(inc["ticket"]["id"], note=f"Executed {step}: {res}")

    inc.setdefault("actions", []).append(res)
    inc["status"] = "ACTIONS_RUN"
    return redirect(url_for("incident", site_id=site_id))
```

---

## 6) HTTP Endpoints

* `GET /` — home page; lists alarms by site
* `POST /ingest` — create/update an incident for a site
* `GET /incident/<site_id>` — incident dashboard
* `POST /triage` — run triage (RAG + LLM)
* `POST /approve` — approve a guarded action (HITL)
* `POST /action` — run a safe action (simulated)
* `GET /api/incident/<site_id>` — JSON view of the incident

---

## 7) Troubleshooting

**ValueError: Circular reference detected**
Fixed by keeping only a **ticket summary** in the incident and avoiding `ticket.details` in the Jinja context.

**Flask port already in use**

```bash
export PORT=5058
./run.sh
```

**Ollama call times out**
Ensure `ollama serve` is running and `OLLAMA_MODEL` matches.

**No triage output?**
Check terminal logs. Use `MOCK_LLM=1` for guaranteed output.

---

## 8) Extending the Demo

* Replace runbook search with **ChromaDB** and embeddings
* Add more KPIs (PRB, RSRP, throughput) and basic charts
* Swap ticket store for SQLite/your ticket API
* Add OpenTelemetry spans + Grafana dashboard

---

## 9) VS Code Debugging

* Run → **Add Configuration…** → Python → Flask
* Set `FLASK_APP=app/orchestrator.py` and `PORT=5057` in the launch config or env

---

## 10) Demo Script (5 minutes)

1. Home → **Ingest →** `DLK123`
2. Incident shows alarms + last syslog lines
3. **Triage ▶** → JSON plan with runbook reference & KPI avg
4. **Run: Load-shed** → ticket opens and logs update
5. **Approve Restart** → **Run: Graceful Restart** → wrap with “KPIs recovered” narrative

