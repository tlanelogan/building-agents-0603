"""Flask web UI for the UdaPlay agent.

Run from the project:

    cd project/starter
    ../../.venv/bin/python webapp/app.py

then open http://127.0.0.1:5000 .

The app wraps a single ``UdaPlayService`` (thread-safe) and exposes a small JSON API
consumed by the vanilla-JS frontend in templates/ + static/.
"""

import os
import sys

# Make the starter/ dir importable so `from lib...` works regardless of cwd.
STARTER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if STARTER_DIR not in sys.path:
    sys.path.insert(0, STARTER_DIR)

from flask import Flask, jsonify, request, render_template

from lib.udaplay_service import UdaPlayService

app = Flask(__name__)

# One shared service for the process. Building it connects to ChromaDB and the agent.
service = UdaPlayService()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/ask", methods=["POST"])
def api_ask():
    data = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()
    if not query:
        return jsonify({"error": "query is required"}), 400
    try:
        result = service.ask(query)
        return jsonify(result)
    except Exception as exc:  # keep the UI alive; surface the error
        return jsonify({"error": str(exc)}), 500


@app.route("/api/history")
def api_history():
    return jsonify(service.history())


@app.route("/api/history/<interaction_id>")
def api_history_item(interaction_id):
    item = service.get(interaction_id)
    if not item:
        return jsonify({"error": "not found"}), 404
    return jsonify(item)


@app.route("/api/knowledge")
def api_knowledge():
    return jsonify(service.knowledge_base())


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
