"""Local web dashboard: talk to Alex from a browser instead of the terminal.

Bound to 127.0.0.1 only (see main()) — Alex has real shell/file access, so
this must never be exposed beyond localhost without deliberately adding
authentication first.
"""
from __future__ import annotations

import threading
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from assistant.orchestrator import Orchestrator

TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"


def create_app(repo_root: Path, cfg: dict) -> Flask:
    app = Flask(__name__, template_folder=str(TEMPLATE_DIR))
    orchestrator = Orchestrator(repo_root, cfg)
    lock = threading.Lock()  # Orchestrator.chat() mutates shared state; serialize requests

    @app.get("/")
    def index():
        name = cfg.get("voice", {}).get("name", "Alex")
        return render_template("index.html", assistant_name=name)

    @app.post("/api/chat")
    def chat():
        data = request.get_json(force=True, silent=True) or {}
        message = (data.get("message") or "").strip()
        if not message:
            return jsonify({"error": "empty message"}), 400
        with lock:
            try:
                reply = orchestrator.chat(message)
            except Exception as exc:
                return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 502
        return jsonify({"reply": reply})

    @app.get("/api/status")
    def status():
        with lock:
            trading_status = orchestrator.registry.execute("trading_bot_status", {})
            proposals = orchestrator.registry.execute("list_proposals", {})
        return jsonify({"trading_bot_status": trading_status, "proposals": proposals})

    return app


def main() -> int:
    import os

    from trading_bot.config import REPO_ROOT, load_env, load_yaml_config

    load_env()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY is not set. Add it to .env (see .env.example).")
        return 1

    cfg = load_yaml_config()
    app = create_app(REPO_ROOT, cfg)
    port = cfg.get("dashboard", {}).get("port", 5000)
    print(f"Dashboard running at http://127.0.0.1:{port} (Ctrl+C to stop)")
    app.run(host="127.0.0.1", port=port, debug=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
