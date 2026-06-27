"""Process-global per-pass progress registry for the R2 watchdog (passive).

The docling-graph batch loop calls start/advance/touch/clear; GET /progress
serves snapshot(). docling-graph is a single uvicorn process but runs up to
DOCLING_GRAPH_MAX_CONCURRENT_EXTRACTIONS passes concurrently on separate
threads, so the shared dict is lock-guarded. No docling_graph imports — pure
stdlib so it loads in standalone unit tests and inside the patched library."""
from __future__ import annotations

import threading
import time

_LOCK = threading.Lock()
# key: (run_id, pass_name) -> {done, total, phase, started_at, updated_at}
_REGISTRY: dict[tuple[str, str], dict] = {}


def start(run_id: str, pass_name: str, total: int) -> None:
    now = time.time()
    with _LOCK:
        _REGISTRY[(run_id, pass_name)] = {
            "done": 0, "total": int(total), "phase": "batches",
            "started_at": now, "updated_at": now,
        }


def advance(run_id: str, pass_name: str) -> None:
    with _LOCK:
        e = _REGISTRY.get((run_id, pass_name))
        if e is not None:
            e["done"] += 1
            e["updated_at"] = time.time()


def touch(run_id: str, pass_name: str, phase: str) -> None:
    with _LOCK:
        e = _REGISTRY.get((run_id, pass_name))
        if e is not None:
            e["phase"] = phase
            e["updated_at"] = time.time()


def clear(run_id: str, pass_name: str) -> None:
    with _LOCK:
        _REGISTRY.pop((run_id, pass_name), None)


def snapshot() -> list[dict]:
    now = time.time()
    with _LOCK:
        return [
            {"run_id": rid, "pass_name": pn, "done": e["done"], "total": e["total"],
             "phase": e["phase"], "started_at": e["started_at"],
             "updated_at": e["updated_at"], "age_s": now - e["updated_at"]}
            for (rid, pn), e in _REGISTRY.items()
        ]
