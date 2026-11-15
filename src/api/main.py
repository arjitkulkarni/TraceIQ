"""Local read-only API over frozen artifacts. It never trains and never invents a score."""
from __future__ import annotations

import json
import os
import sqlite3
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.common import ARTIFACTS, ROOT, load_config

DB_PATH = Path(os.environ.get("TRACEIQ_DB", ARTIFACTS / "traceiq.db"))
FRONTEND_DIST = ROOT / "frontend" / "dist"
DISCLAIMER = (
    "Model evidence for analyst prioritization only. A connected suspicious transaction does not "
    "imply the focal transaction is criminal."
)

cfg = load_config()["serving"]
app = FastAPI(title="TraceIQ local API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    """Read-only connection, closed on exit (sqlite3's own context manager does not close)."""
    if not DB_PATH.exists():
        raise HTTPException(503, "Index not built. Run python -m src.api.build_index.")
    connection = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
    finally:
        connection.close()


def meta(connection: sqlite3.Connection, key: str):
    row = connection.execute("SELECT payload FROM meta WHERE key = ?", (key,)).fetchone()
    return json.loads(row["payload"]) if row else None


def public_transaction(row: sqlite3.Row, serving: dict) -> dict:
    return {
        "transaction_id": row["tx_id"],
        "time_step": row["time_step"],
        "split": row["split"],
        "dataset_label": row["dataset_label"],
        "model_name": serving["model_name"],
        "model_score": row["model_score"],
        "score_type": serving["score_type"],
        "risk_tier": row["risk_tier"],
        "explanation_available": bool(row["explanation_available"]),
    }


def get_row(connection: sqlite3.Connection, tx_id: str) -> sqlite3.Row:
    row = connection.execute("SELECT * FROM transactions WHERE tx_id = ?", (tx_id,)).fetchone()
    if row is None:
        raise HTTPException(404, f"Transaction {tx_id} is not in this dataset.")
    return row


@app.get("/api/health")
@app.get("/health")
def health() -> dict:
    if not DB_PATH.exists():
        return {"status": "not_ready", "reason": "index not built"}
    with connect() as connection:
        serving = meta(connection, "serving")
        count = connection.execute("SELECT COUNT(*) AS n FROM transactions").fetchone()["n"]
    return {"status": "ok", "transactions_indexed": count, "model": serving["model_name"], "run": serving["run"]}


@app.get("/api/metrics")
@app.get("/metrics")
def metrics() -> dict:
    with connect() as connection:
        rows = connection.execute("SELECT key, payload FROM meta").fetchall()
    return {row["key"]: json.loads(row["payload"]) for row in rows}


@app.get("/api/overview")
def overview() -> dict:
    with connect() as connection:
        total = connection.execute("SELECT COUNT(*) AS n FROM transactions").fetchone()["n"]
        by_label = connection.execute(
            "SELECT dataset_label AS k, COUNT(*) AS n FROM transactions GROUP BY 1").fetchall()
        by_tier = connection.execute(
            "SELECT risk_tier AS k, COUNT(*) AS n FROM transactions WHERE split = 'test' GROUP BY 1").fetchall()
        test_total = connection.execute(
            "SELECT COUNT(*) AS n FROM transactions WHERE split = 'test'").fetchone()["n"]
        steps = connection.execute(
            """SELECT time_step, COUNT(*) AS transactions,
                      SUM(risk_tier = 'high') AS flagged,
                      SUM(dataset_label = 'illicit') AS known_illicit
               FROM transactions GROUP BY time_step ORDER BY time_step""").fetchall()
        serving = meta(connection, "serving")
        explained = connection.execute("SELECT COUNT(*) AS n FROM explanations").fetchone()["n"]
    return {
        "transactions_scored": total,
        "by_label": {row["k"]: row["n"] for row in by_label},
        "test_period_transactions": test_total,
        "test_period_by_tier": {row["k"]: row["n"] for row in by_tier},
        "steps": [dict(row) for row in steps],
        "serving": serving,
        "explanations_available": explained,
    }


@app.get("/api/transactions")
@app.get("/transactions")
def transactions(
    sort: Literal["risk", "time"] = "risk",
    page: int = Query(1, ge=1),
    page_size: int = Query(cfg["default_page_size"], ge=1),
    tier: Literal["high", "medium", "low"] | None = None,
    label: Literal["illicit", "licit", "unknown"] | None = None,
    split: Literal["train", "val", "test"] | None = None,
    time_step: int | None = Query(None, ge=1, le=49),
    explained: bool | None = None,
) -> dict:
    page_size = min(page_size, cfg["max_page_size"])
    clauses, values = [], []
    for column, value in (("risk_tier", tier), ("dataset_label", label), ("split", split), ("time_step", time_step)):
        if value is not None:
            clauses.append(f"{column} = ?")
            values.append(value)
    if explained is not None:
        clauses.append("explanation_available = ?")
        values.append(int(explained))
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    order = "model_score DESC, node_index" if sort == "risk" else "time_step, model_score DESC, node_index"
    with connect() as connection:
        serving = meta(connection, "serving")
        total = connection.execute(f"SELECT COUNT(*) AS n FROM transactions {where}", values).fetchone()["n"]
        rows = connection.execute(
            f"SELECT * FROM transactions {where} ORDER BY {order} LIMIT ? OFFSET ?",
            [*values, page_size, (page - 1) * page_size],
        ).fetchall()
    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "items": [public_transaction(row, serving) for row in rows],
    }


@app.get("/api/transactions/{tx_id}")
@app.get("/transactions/{tx_id}")
def transaction(tx_id: str) -> dict:
    with connect() as connection:
        serving = meta(connection, "serving")
        row = get_row(connection, tx_id)
    return {
        **public_transaction(row, serving),
        "alert_threshold": serving["threshold"],
        "run": serving["run"],
        "in_degree": row["in_degree"],
        "out_degree": row["out_degree"],
        "tabular_comparison_score": row["tabular_score"],
        "anomaly_percentile": row["anomaly_percentile"],
        "disclaimer": DISCLAIMER,
    }


def bounded_neighborhood(connection, focal: int, hops: int, max_nodes: int) -> tuple[list[int], bool]:
    """Breadth-first walk over both edge directions, capped. Nearest nodes are kept first."""
    seen = {focal: 0}
    queue = deque([focal])
    truncated = False
    while queue:
        node = queue.popleft()
        depth = seen[node]
        if depth == hops:
            continue
        neighbors = connection.execute(
            "SELECT dst AS n FROM edges WHERE src = ? UNION SELECT src AS n FROM edges WHERE dst = ? ORDER BY n",
            (node, node),
        ).fetchall()
        for row in neighbors:
            if row["n"] in seen:
                continue
            if len(seen) >= max_nodes:
                truncated = True
                break
            seen[row["n"]] = depth + 1
            queue.append(row["n"])
    return list(seen), truncated


@app.get("/api/transactions/{tx_id}/graph")
@app.get("/transactions/{tx_id}/graph")
def transaction_graph(
    tx_id: str,
    hops: int = Query(1, ge=1),
    max_nodes: int = Query(cfg["graph_max_nodes"], ge=2),
) -> dict:
    hops = min(hops, cfg["graph_max_hops"])
    max_nodes = min(max_nodes, cfg["graph_max_nodes"])
    with connect() as connection:
        serving = meta(connection, "serving")
        focal = get_row(connection, tx_id)
        members, truncated = bounded_neighborhood(connection, focal["node_index"], hops, max_nodes)
        marks = ",".join("?" * len(members))
        rows = connection.execute(f"SELECT * FROM transactions WHERE node_index IN ({marks})", members).fetchall()
        edge_rows = connection.execute(
            f"SELECT src, dst FROM edges WHERE src IN ({marks}) AND dst IN ({marks})", [*members, *members]
        ).fetchall()
        explanation = connection.execute("SELECT payload FROM explanations WHERE tx_id = ?", (tx_id,)).fetchone()
    tx_of = {row["node_index"]: row["tx_id"] for row in rows}
    evidence = {}
    if explanation:
        evidence = {(e["source"], e["target"]): e for e in json.loads(explanation["payload"])["edges"]}
    return {
        "focal": tx_id,
        "time_step": focal["time_step"],
        "hops": hops,
        "max_nodes": max_nodes,
        "truncated": truncated,
        "nodes": [
            {**public_transaction(row, serving), "is_focal": row["tx_id"] == tx_id,
             "in_degree": row["in_degree"], "out_degree": row["out_degree"]}
            for row in rows
        ],
        "edges": [
            {"source": tx_of[e["src"]], "target": tx_of[e["dst"]],
             "importance": evidence.get((tx_of[e["src"]], tx_of[e["dst"]]), {}).get("importance"),
             "logit_drop": evidence.get((tx_of[e["src"]], tx_of[e["dst"]]), {}).get("occlusion_logit_drop")}
            for e in edge_rows
        ],
    }


@app.get("/api/transactions/{tx_id}/explanation")
@app.get("/transactions/{tx_id}/explanation")
def transaction_explanation(tx_id: str) -> dict:
    with connect() as connection:
        serving = meta(connection, "serving")
        get_row(connection, tx_id)
        row = connection.execute("SELECT payload FROM explanations WHERE tx_id = ?", (tx_id,)).fetchone()
    if row is None:
        return {"available": False, "model_name": serving["model_name"], "disclaimer": DISCLAIMER}
    return {"available": True, **json.loads(row["payload"]), "disclaimer": DISCLAIMER}


if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def frontend(path: str) -> FileResponse:
        if path.startswith("api/"):
            raise HTTPException(404, "Unknown API route.")
        return FileResponse(FRONTEND_DIST / "index.html")
