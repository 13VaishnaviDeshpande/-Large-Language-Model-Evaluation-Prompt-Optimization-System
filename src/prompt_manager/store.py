"""
Prompt Store — Enhanced
SQLite-backed storage with batch operations, CSV import/export,
pagination for 200+ variants, and per-variant analytics.
"""

import sqlite3
import json
import uuid
import csv
import io
import os
from pathlib import Path
from datetime import datetime
from typing import Optional

DB_PATH = Path(__file__).parent.parent.parent / "data" / "eval_store.db"


def _get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def init_db():
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS prompts (
            id          TEXT PRIMARY KEY,
            name        TEXT NOT NULL,
            text        TEXT NOT NULL,
            tags        TEXT DEFAULT '[]',
            group_id    TEXT DEFAULT NULL,
            source      TEXT DEFAULT 'manual',
            created_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL,
            version     INTEGER DEFAULT 1,
            parent_id   TEXT DEFAULT NULL
        );

        CREATE TABLE IF NOT EXISTS eval_results (
            run_id              TEXT PRIMARY KEY,
            prompt_id           TEXT NOT NULL,
            prompt_name         TEXT,
            prompt_text         TEXT,
            task_description    TEXT,
            reference_answer    TEXT,
            llm_response        TEXT,
            model               TEXT,
            temperature         REAL,
            bleu                REAL,
            rouge_l             REAL,
            semantic_similarity REAL,
            coherence           REAL,
            faithfulness        REAL,
            completeness        REAL,
            clarity             REAL,
            usefulness          REAL,
            judge_reasoning     TEXT,
            judge_used          INTEGER,
            composite           REAL,
            latency_ms          INTEGER,
            timestamp           TEXT,
            error               TEXT,
            batch_id            TEXT DEFAULT NULL,
            csv_row             INTEGER DEFAULT NULL
        );

        CREATE TABLE IF NOT EXISTS batch_runs (
            batch_id        TEXT PRIMARY KEY,
            task_description TEXT,
            reference_answer TEXT,
            model           TEXT,
            temperature     REAL,
            total_variants  INTEGER,
            completed       INTEGER DEFAULT 0,
            started_at      TEXT,
            finished_at     TEXT,
            source          TEXT DEFAULT 'manual'
        );

        CREATE INDEX IF NOT EXISTS idx_results_prompt_id  ON eval_results(prompt_id);
        CREATE INDEX IF NOT EXISTS idx_results_timestamp  ON eval_results(timestamp);
        CREATE INDEX IF NOT EXISTS idx_results_composite  ON eval_results(composite DESC);
        CREATE INDEX IF NOT EXISTS idx_results_batch_id   ON eval_results(batch_id);
        CREATE INDEX IF NOT EXISTS idx_prompts_group_id   ON prompts(group_id);
    """)
    conn.commit()
    conn.close()


# ── Prompt CRUD ────────────────────────────────────────────────────────────────

def save_prompt(name: str, text: str, tags: list = None,
                parent_id: str = None, group_id: str = None,
                source: str = "manual") -> str:
    init_db()
    pid = str(uuid.uuid4())[:12]
    now = datetime.now().isoformat()
    conn = _get_conn()
    conn.execute(
        """INSERT INTO prompts (id, name, text, tags, created_at, updated_at,
                                parent_id, group_id, source)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (pid, name, text, json.dumps(tags or []), now, now, parent_id, group_id, source),
    )
    conn.commit()
    conn.close()
    return pid


def save_prompts_bulk(prompts: list, group_id: str = None, source: str = "csv") -> list:
    """Insert many prompts in one transaction. Returns list of IDs."""
    init_db()
    now = datetime.now().isoformat()
    gid = group_id or str(uuid.uuid4())[:8]
    ids = []
    conn = _get_conn()
    for p in prompts:
        pid = str(uuid.uuid4())[:12]
        conn.execute(
            """INSERT INTO prompts (id, name, text, tags, created_at, updated_at,
                                    parent_id, group_id, source)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (pid, p.get("name", f"Prompt {pid}"),
             p.get("text", ""), json.dumps(p.get("tags", [])),
             now, now, p.get("parent_id"), gid, source),
        )
        ids.append(pid)
    conn.commit()
    conn.close()
    return ids


def update_prompt(prompt_id: str, name: str = None, text: str = None, tags: list = None):
    init_db()
    conn = _get_conn()
    row = conn.execute("SELECT * FROM prompts WHERE id=?", (prompt_id,)).fetchone()
    if not row:
        conn.close()
        raise ValueError(f"Prompt {prompt_id} not found")
    new_name    = name or row["name"]
    new_text    = text or row["text"]
    new_tags    = json.dumps(tags) if tags is not None else row["tags"]
    new_version = row["version"] + 1
    now = datetime.now().isoformat()
    conn.execute(
        "UPDATE prompts SET name=?, text=?, tags=?, updated_at=?, version=? WHERE id=?",
        (new_name, new_text, new_tags, now, new_version, prompt_id),
    )
    conn.commit()
    conn.close()


def get_prompt(prompt_id: str) -> Optional[dict]:
    init_db()
    conn = _get_conn()
    row = conn.execute("SELECT * FROM prompts WHERE id=?", (prompt_id,)).fetchone()
    conn.close()
    if not row:
        return None
    d = dict(row)
    d["tags"] = json.loads(d["tags"])
    return d


def list_prompts(limit: int = 500, offset: int = 0, group_id: str = None) -> list:
    init_db()
    conn = _get_conn()
    if group_id:
        rows = conn.execute(
            "SELECT * FROM prompts WHERE group_id=? ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (group_id, limit, offset),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM prompts ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
    conn.close()
    result = []
    for r in rows:
        d = dict(r)
        d["tags"] = json.loads(d["tags"])
        result.append(d)
    return result


def count_prompts(group_id: str = None) -> int:
    init_db()
    conn = _get_conn()
    if group_id:
        n = conn.execute("SELECT COUNT(*) FROM prompts WHERE group_id=?",
                         (group_id,)).fetchone()[0]
    else:
        n = conn.execute("SELECT COUNT(*) FROM prompts").fetchone()[0]
    conn.close()
    return n


def delete_prompt(prompt_id: str):
    init_db()
    conn = _get_conn()
    conn.execute("DELETE FROM prompts WHERE id=?", (prompt_id,))
    conn.execute("DELETE FROM eval_results WHERE prompt_id=?", (prompt_id,))
    conn.commit()
    conn.close()


def delete_group(group_id: str):
    init_db()
    conn = _get_conn()
    pids = [r[0] for r in conn.execute(
        "SELECT id FROM prompts WHERE group_id=?", (group_id,)).fetchall()]
    if pids:
        placeholders = ",".join("?" * len(pids))
        conn.execute(f"DELETE FROM eval_results WHERE prompt_id IN ({placeholders})", pids)
    conn.execute("DELETE FROM prompts WHERE group_id=?", (group_id,))
    conn.commit()
    conn.close()


# ── CSV Import / Export ────────────────────────────────────────────────────────

def import_prompts_from_csv(csv_content: str):
    """
    Parse CSV into prompt dicts. Returns (rows, errors).
    Accepted columns (case-insensitive):
      name / prompt_name / variant_name  → name
      text / prompt / prompt_text        → text
      tags                               → tags (comma-separated within cell)
    """
    rows, errors = [], []
    try:
        reader = csv.DictReader(io.StringIO(csv_content.strip()))
        col_map = {}
        if reader.fieldnames:
            for fn in reader.fieldnames:
                fl = fn.strip().lower()
                if fl in ("name", "prompt_name", "variant_name", "variant"):
                    col_map["name"] = fn
                elif fl in ("text", "prompt", "prompt_text", "content", "template"):
                    col_map["text"] = fn
                elif fl in ("tags", "tag", "category", "categories"):
                    col_map["tags"] = fn

        if "text" not in col_map:
            errors.append("CSV must have a column named 'text', 'prompt', or 'prompt_text'.")
            return rows, errors

        for i, row in enumerate(reader, start=2):
            text = row.get(col_map["text"], "").strip()
            if not text:
                errors.append(f"Row {i}: empty prompt text — skipped.")
                continue
            name_col = col_map.get("name", "")
            name = (row.get(name_col, "") or "").strip() or f"Variant {i-1}"
            raw_tags = row.get(col_map.get("tags", ""), "") or ""
            tags = [t.strip() for t in raw_tags.split(",") if t.strip()]
            rows.append({"name": name, "text": text, "tags": tags})

    except Exception as e:
        errors.append(f"CSV parse error: {e}")

    return rows, errors


def export_results_to_csv(results: list) -> str:
    if not results:
        return ""
    fieldnames = list(results[0].keys())
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)
    return buf.getvalue()


# ── Batch Run Tracking ─────────────────────────────────────────────────────────

def create_batch(task_description: str, reference_answer: str,
                 model: str, temperature: float,
                 total_variants: int, source: str = "manual") -> str:
    init_db()
    bid = str(uuid.uuid4())[:10]
    now = datetime.now().isoformat()
    conn = _get_conn()
    conn.execute(
        """INSERT INTO batch_runs (batch_id, task_description, reference_answer, model,
                                   temperature, total_variants, started_at, source)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (bid, task_description, reference_answer, model, temperature,
         total_variants, now, source),
    )
    conn.commit()
    conn.close()
    return bid


def update_batch_progress(batch_id: str, completed: int, finished: bool = False):
    init_db()
    conn = _get_conn()
    if finished:
        conn.execute(
            "UPDATE batch_runs SET completed=?, finished_at=? WHERE batch_id=?",
            (completed, datetime.now().isoformat(), batch_id),
        )
    else:
        conn.execute("UPDATE batch_runs SET completed=? WHERE batch_id=?",
                     (completed, batch_id))
    conn.commit()
    conn.close()


def list_batches(limit: int = 50) -> list:
    init_db()
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM batch_runs ORDER BY started_at DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── Eval Result Storage ────────────────────────────────────────────────────────

def save_result(result, batch_id: str = None, csv_row: int = None) -> None:
    init_db()
    d = result.to_dict()
    d["batch_id"] = batch_id
    d["csv_row"] = csv_row
    conn = _get_conn()
    conn.execute(
        """INSERT OR REPLACE INTO eval_results
           (run_id, prompt_id, prompt_name, prompt_text, task_description,
            reference_answer, llm_response, model, temperature,
            bleu, rouge_l, semantic_similarity, coherence,
            faithfulness, completeness, clarity, usefulness,
            judge_reasoning, judge_used, composite, latency_ms, timestamp, error,
            batch_id, csv_row)
           VALUES
           (:run_id, :prompt_id, :prompt_name, :prompt_text, :task_description,
            :reference_answer, :llm_response, :model, :temperature,
            :bleu, :rouge_l, :semantic_similarity, :coherence,
            :faithfulness, :completeness, :clarity, :usefulness,
            :judge_reasoning, :judge_used, :composite, :latency_ms, :timestamp, :error,
            :batch_id, :csv_row)""",
        d,
    )
    conn.commit()
    conn.close()


def save_results_bulk(results: list, batch_id: str = None) -> None:
    """Bulk-insert eval results in one transaction."""
    init_db()
    conn = _get_conn()
    for i, result in enumerate(results):
        d = result.to_dict()
        d["batch_id"] = batch_id
        d["csv_row"] = i
        conn.execute(
            """INSERT OR REPLACE INTO eval_results
               (run_id, prompt_id, prompt_name, prompt_text, task_description,
                reference_answer, llm_response, model, temperature,
                bleu, rouge_l, semantic_similarity, coherence,
                faithfulness, completeness, clarity, usefulness,
                judge_reasoning, judge_used, composite, latency_ms, timestamp, error,
                batch_id, csv_row)
               VALUES
               (:run_id, :prompt_id, :prompt_name, :prompt_text, :task_description,
                :reference_answer, :llm_response, :model, :temperature,
                :bleu, :rouge_l, :semantic_similarity, :coherence,
                :faithfulness, :completeness, :clarity, :usefulness,
                :judge_reasoning, :judge_used, :composite, :latency_ms, :timestamp, :error,
                :batch_id, :csv_row)""",
            d,
        )
    conn.commit()
    conn.close()


def get_results(prompt_id: str = None, batch_id: str = None,
                limit: int = 500, offset: int = 0) -> list:
    init_db()
    conn = _get_conn()
    if prompt_id:
        rows = conn.execute(
            "SELECT * FROM eval_results WHERE prompt_id=? ORDER BY timestamp DESC LIMIT ? OFFSET ?",
            (prompt_id, limit, offset),
        ).fetchall()
    elif batch_id:
        rows = conn.execute(
            "SELECT * FROM eval_results WHERE batch_id=? ORDER BY composite DESC LIMIT ? OFFSET ?",
            (batch_id, limit, offset),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM eval_results ORDER BY timestamp DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_leaderboard(limit: int = 500) -> list:
    init_db()
    conn = _get_conn()
    rows = conn.execute(
        """SELECT * FROM eval_results
           WHERE (error = '' OR error IS NULL)
           ORDER BY composite DESC LIMIT ?""",
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_stats() -> dict:
    init_db()
    conn = _get_conn()
    stats = conn.execute(
        """SELECT COUNT(*) as total_runs,
                  COUNT(DISTINCT prompt_id) as unique_prompts,
                  AVG(composite) as avg_composite,
                  MAX(composite) as best_score,
                  MIN(composite) as worst_score,
                  AVG(latency_ms) as avg_latency_ms
           FROM eval_results WHERE error='' OR error IS NULL"""
    ).fetchone()
    conn.close()
    return dict(stats) if stats else {}


def get_variant_analysis(batch_id: str) -> list:
    """Per-variant deep analysis for a specific batch run."""
    init_db()
    conn = _get_conn()
    rows = conn.execute(
        """SELECT run_id, prompt_name, composite, bleu, rouge_l,
                  semantic_similarity, coherence, faithfulness,
                  completeness, clarity, usefulness,
                  latency_ms, judge_reasoning, llm_response,
                  prompt_text, error, csv_row
           FROM eval_results
           WHERE batch_id=?
           ORDER BY composite DESC""",
        (batch_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
