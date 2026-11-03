"""scripts/cloud_loop.py racing_now(): faster ticks while racing; never crashes the loop."""
import importlib.util
import sqlite3
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("cloud_loop", ROOT / "scripts" / "cloud_loop.py")
cloud_loop = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cloud_loop)


def _db(tmp_path, posts, date=None):
    p = tmp_path / "t.db"
    c = sqlite3.connect(p)
    c.execute("CREATE TABLE races (race_id TEXT, date TEXT, post_time TEXT)")
    for i, t in enumerate(posts, 1):
        c.execute("INSERT INTO races VALUES (?,?,?)", (f"r{i}", date or (t[:10] if t else "2026-11-04"), t))
    c.commit()
    c.close()
    return str(p)


def test_racing_window(tmp_path):
    db = _db(tmp_path, ["2026-11-04T18:40", "2026-11-04T22:50"])
    at = lambda s: cloud_loop.racing_now(db, datetime.strptime(s, "%Y-%m-%d %H:%M"))
    assert not at("2026-11-04 18:29")
    assert at("2026-11-04 18:30")          # 10 min before race 1
    assert at("2026-11-04 20:00")
    assert at("2026-11-04 23:50")          # last race + 1 h (inclusive)
    assert not at("2026-11-04 23:51")
    assert not at("2026-11-05 12:00")      # another day


def test_racing_now_never_raises(tmp_path):
    assert cloud_loop.racing_now(str(tmp_path / "missing.db")) is False
    assert not (tmp_path / "missing.db").exists()            # read-only: no empty DB created
    bad = _db(tmp_path, ["garbage"], date="2026-11-04")       # row found, strptime fails -> False
    assert cloud_loop.racing_now(bad, datetime(2026, 11, 4, 20, 0)) is False
