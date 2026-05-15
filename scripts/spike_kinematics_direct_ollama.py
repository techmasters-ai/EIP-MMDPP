"""Single-batch Gemma4 spike — sends the post-edit patched system + user prompt
for batch_06 (the SA-2 table batch with 4 missile variants) directly to Ollama
gemma4:31b and inspects whether numeric kinematic fields get filled.

Baseline (pre-edit) for batch 06: 4 entities / 0 filled properties (Gemma4 + Claude).
Target (post-edit, simulated by Claude): 4 entities / 13 filled properties (65%).

This spike is the model-side confirmation: does the prompt change unlock
numerics for Gemma4 as well?
"""
from __future__ import annotations

import json
import re
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

WORK = Path(__file__).resolve().parent.parent / "tmp" / "claude_vs_gemma4"
SYS_FILE = WORK / "batch_06_MissileKinematicsPass_58_of_58_patched_system.txt"
USR_FILE = WORK / "batch_06_MissileKinematicsPass_58_of_58_patched_user.txt"

# Hit Ollama via the docling-graph container so we use the same network path
# that production extraction uses. Run via:
#   docker exec eip-mmdpp-docling-graph-1 python3 /app/spike_kinematics_direct_ollama.py
# OR: change OLLAMA_BASE to localhost:11434 if ollama is exposed.
OLLAMA_BASE = "http://ollama:11434"
MODEL = "gemma4:31b"


def main() -> int:
    if not SYS_FILE.exists() or not USR_FILE.exists():
        print(f"[error] patched prompt files not found at {WORK}")
        return 1

    sys_msg = SYS_FILE.read_text()
    usr_msg = USR_FILE.read_text()

    print(f"[info] system bytes={len(sys_msg)}, user bytes={len(usr_msg)}")
    has_unit_hint = "UNITS: Numeric values" in usr_msg
    has_strict_gate = "value AND unit" in usr_msg
    has_relaxed_macro = "When the source value AND its" in sys_msg
    print(f"[info] user has UNIT_HINT preamble: {has_unit_hint}")
    print(f"[info] user has strict gate phrase: {has_strict_gate} (should be False)")
    print(f"[info] system has relaxed macro:   {has_relaxed_macro} (should be True)")

    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": usr_msg},
        ],
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_ctx": 56000,
        },
        "format": "json",
    }
    print(f"[info] calling {OLLAMA_BASE}/api/chat (model={MODEL}) …", flush=True)
    t0 = time.monotonic()
    try:
        req = urllib.request.Request(
            f"{OLLAMA_BASE}/api/chat",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=900) as r:
            response = json.loads(r.read())
    except urllib.error.HTTPError as e:
        msg = e.read().decode()[:1500]
        print(f"[error] HTTP {e.code}: {msg}")
        return 1
    elapsed = time.monotonic() - t0

    msg = response.get("message") or {}
    content = msg.get("content", "")
    print(f"[ok] elapsed={elapsed:.1f}s, content bytes={len(content)}")

    # Parse JSON (may be fenced)
    cleaned = re.sub(r'^```(?:json)?\s*', '', content.strip())
    cleaned = re.sub(r'\s*```$', '', cleaned)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        print(f"[error] JSON parse failed: {e}")
        print(f"\n--- Gemma4 raw content ---\n{content[:3000]}")
        return 2

    nodes = parsed.get("nodes", [])
    missiles = [n for n in nodes if n.get("path") == "missile_systems[]"]
    field_keys = [
        "min_intercept_km", "max_intercept_km",
        "min_altitude_km", "max_altitude_km",
        "max_launch_angle_deg",
    ]
    n_filled = 0
    for m in missiles:
        props = m.get("properties", {}) or {}
        for k in field_keys:
            if props.get(k) is not None:
                n_filled += 1

    print(f"\n[ok] entities={len(missiles)}, filled={n_filled}/{len(missiles) * len(field_keys)}")
    print(f"[ok] avg_fill = {n_filled / max(len(missiles), 1):.2f}/{len(field_keys)}")

    print("\n=== ENTITIES + KINEMATIC FIELDS ===")
    for m in missiles:
        row = {k: (m.get("properties") or {}).get(k) for k in field_keys}
        print(f"  {(m.get('ids') or {}).get('system_name'):<10} {row}")

    out = WORK / "spike_gemma4_post_edit_batch06_response.json"
    out.write_text(json.dumps(response, indent=2, ensure_ascii=False))
    print(f"\n[ok] saved full response to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
