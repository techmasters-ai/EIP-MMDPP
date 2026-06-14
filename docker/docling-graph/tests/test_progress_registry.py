import importlib.util, threading, time
from pathlib import Path

_MOD = Path(__file__).resolve().parent.parent / "app" / "_progress_registry.py"
_spec = importlib.util.spec_from_file_location("dg_progress_registry", _MOD)
reg = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(reg)


def setup_function(_):
    reg._REGISTRY.clear()


def test_start_advance_clear():
    reg.start("run1", "radar_identity", total=3)
    snap = {(s["run_id"], s["pass_name"]): s for s in reg.snapshot()}
    assert snap[("run1", "radar_identity")]["done"] == 0
    assert snap[("run1", "radar_identity")]["total"] == 3
    reg.advance("run1", "radar_identity")
    reg.advance("run1", "radar_identity")
    assert {(s["run_id"], s["pass_name"]): s["done"] for s in reg.snapshot()}[("run1", "radar_identity")] == 2
    reg.clear("run1", "radar_identity")
    assert reg.snapshot() == []


def test_touch_bumps_updated_at_not_done():
    reg.start("r", "p", total=5)
    reg.advance("r", "p")
    before = [s for s in reg.snapshot() if s["pass_name"] == "p"][0]
    time.sleep(0.01)
    reg.touch("r", "p", phase="merging")
    after = [s for s in reg.snapshot() if s["pass_name"] == "p"][0]
    assert after["done"] == before["done"] == 1
    assert after["updated_at"] > before["updated_at"]
    assert after["phase"] == "merging"


def test_two_keys_isolated_under_threads():
    def work(rid):
        reg.start(rid, "p", total=50)
        for _ in range(50):
            reg.advance(rid, "p")
    t1 = threading.Thread(target=work, args=("a",)); t2 = threading.Thread(target=work, args=("b",))
    t1.start(); t2.start(); t1.join(); t2.join()
    done = {(s["run_id"]): s["done"] for s in reg.snapshot()}
    assert done["a"] == 50 and done["b"] == 50


def test_age_s_monotonic():
    reg.start("r", "p", total=1)
    time.sleep(0.02)
    assert [s for s in reg.snapshot()][0]["age_s"] >= 0.02
