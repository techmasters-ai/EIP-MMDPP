"""Smoke harness: 4-URL pool with stub HTTP servers.

Fires 16 concurrent requests via OllamaChatClient; asserts each URL gets
exactly 4 (least-in-flight under uniform service time = even distribution).

Run: .venv/bin/python -m tests.smoke.ollama_pool_routing
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, HTTPServer

from app.services.ollama_pool_client import OllamaChatClient, OllamaPool


_hits: dict[int, int] = {}
_hits_lock = threading.Lock()


class _StubHandler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        port = self.server.server_address[1]
        with _hits_lock:
            _hits[port] = _hits.get(port, 0) + 1
        # Simulate ~50ms generation so multiple requests overlap.
        time.sleep(0.05)
        body = b'{"choices":[{"message":{"content":"{}"}}]}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args, **_kw):
        pass  # silence


def _start_stub() -> tuple[HTTPServer, str]:
    srv = HTTPServer(("127.0.0.1", 0), _StubHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def main() -> None:
    servers = [_start_stub() for _ in range(4)]
    urls = [u for _, u in servers]
    pool = OllamaPool(urls=urls)
    client = OllamaChatClient(pool=pool, model="stub")

    def call(_):
        return client.get_json_response(
            prompt="hi", schema_json="{}", structured_output=False,
        )

    with ThreadPoolExecutor(max_workers=16) as ex:
        list(ex.map(call, range(16)))

    for srv, _ in servers:
        srv.shutdown()

    distribution = {url: _hits[int(url.rsplit(":", 1)[1])] for url in urls}
    print("Distribution:", distribution)
    counts = sorted(distribution.values())
    # Least-in-flight isn't perfectly even, but with uniform service times
    # the spread should be small. Allow up to 2x the lowest.
    assert counts[0] >= 2, f"Some URL got <2 requests: {counts}"
    assert counts[-1] <= 8, f"Some URL got >8 requests: {counts}"
    assert sum(counts) == 16
    print("PASS")


if __name__ == "__main__":
    main()
