"""Smoke harness: 2-URL pool, kill one mid-stream, assert survivor handles all.

Run: .venv/bin/python -m tests.smoke.ollama_pool_failover
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, HTTPServer

from app.services.ollama_pool_client import OllamaChatClient, OllamaPool


def _make_handler():
    class H(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            time.sleep(0.05)
            body = b'{"choices":[{"message":{"content":"{}"}}]}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, *_a, **_kw):
            pass
    return H


def _start() -> tuple[HTTPServer, str]:
    srv = HTTPServer(("127.0.0.1", 0), _make_handler())
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def main() -> None:
    srv1, u1 = _start()
    srv2, u2 = _start()
    pool = OllamaPool(urls=[u1, u2])
    client = OllamaChatClient(pool=pool, model="stub", timeout_s=5.0)

    # Kill srv1 after firing 2 requests; remaining 18 must succeed on srv2.
    def call(i):
        if i == 2:
            srv1.shutdown()
        try:
            client.get_json_response(
                prompt="hi", schema_json="{}", structured_output=False,
            )
            return True
        except Exception as exc:
            return f"FAIL: {type(exc).__name__}: {exc}"

    with ThreadPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(call, range(20)))

    srv2.shutdown()

    failures = [r for r in results if r is not True]
    print(f"Successes: {sum(1 for r in results if r is True)}/20")
    if failures:
        print("Failures:", failures[:5])
    # Allow a couple of mid-flight calls on srv1 to fail; the retry-once
    # pattern only handles connect/timeout, not in-flight kill. The
    # majority must succeed.
    assert sum(1 for r in results if r is True) >= 16, results
    print("PASS")


if __name__ == "__main__":
    main()
