"""SQLite schema and helpers. See SPEC.md for conventions.

Environment overrides (for dry runs / tests on a copy, never touching the real data):
    HKJC_DB=/path/copy.db   database used by connect() when no path is given
"""
import os
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = ROOT / "data" / "hkjc.db"


def db_path() -> Path:
    """The database connect() uses by default: $HKJC_DB if set, else data/hkjc.db."""
    env = os.environ.get("HKJC_DB")
    return Path(env).expanduser() if env else DEFAULT_DB

SCHEMA = """
CREATE TABLE IF NOT EXISTS meetings (
    date TEXT NOT NULL,              -- YYYY-MM-DD
    venue TEXT NOT NULL,             -- ST / HV
    n_races INTEGER,
    first_post_time TEXT,            -- YYYY-MM-DDTHH:MM HK time
    status TEXT,                     -- scheduled / card / results
    updated_at TEXT,
    PRIMARY KEY (date, venue)
);

CREATE TABLE IF NOT EXISTS races (
    race_id TEXT PRIMARY KEY,        -- YYYYMMDD-VV-NN
    date TEXT NOT NULL,
    venue TEXT NOT NULL,
    race_no INTEGER NOT NULL,
    season_race_index INTEGER,       -- the "(99)" number on results page
    race_class TEXT,                 -- e.g. 第五班 / 二級賽 / 新馬賽
    rating_band TEXT,                -- e.g. 40-0
    distance INTEGER,                -- metres
    surface TEXT,                    -- TURF / AWT
    course TEXT,                     -- e.g. A, C+3
    going TEXT,                      -- 場地狀況 e.g. 好地
    race_name TEXT,
    prize INTEGER,
    post_time TEXT,                  -- YYYY-MM-DDTHH:MM
    field_size INTEGER,
    winning_time_s REAL,
    sectional_times TEXT             -- JSON list
);
CREATE INDEX IF NOT EXISTS idx_races_date ON races(date);

CREATE TABLE IF NOT EXISTS runners (
    race_id TEXT NOT NULL,
    horse_no INTEGER NOT NULL,
    horse_id TEXT,                   -- brand no, e.g. K130
    horse_name TEXT,
    jockey_id TEXT,
    jockey_name TEXT,
    trainer_id TEXT,
    trainer_name TEXT,
    act_weight INTEGER,
    body_weight INTEGER,
    draw INTEGER,
    finish_pos INTEGER,              -- NULL if not finished / scratched
    finish_status TEXT,              -- raw placing text, e.g. 1, 2 DH, WV, PU, DNF
    lbw TEXT,
    lbw_num REAL,                    -- lengths behind winner (0 for winner)
    running_pos TEXT,                -- space separated
    finish_time_s REAL,
    win_odds REAL,                   -- final win odds
    scratched INTEGER DEFAULT 0,
    PRIMARY KEY (race_id, horse_no)
);
CREATE INDEX IF NOT EXISTS idx_runners_horse ON runners(horse_id);
CREATE INDEX IF NOT EXISTS idx_runners_jockey ON runners(jockey_id);
CREATE INDEX IF NOT EXISTS idx_runners_trainer ON runners(trainer_id);

CREATE TABLE IF NOT EXISTS dividends (
    race_id TEXT NOT NULL,           -- for multi-race pools: race_id of the last leg
    pool TEXT NOT NULL,              -- WIN PLA QIN QPL FCT TCE TRI FF QTT DBL ...
    combination TEXT NOT NULL,
    dividend REAL,                   -- HK$ per $10
    PRIMARY KEY (race_id, pool, combination)
);

CREATE TABLE IF NOT EXISTS racecard (
    date TEXT NOT NULL,
    venue TEXT NOT NULL,
    race_no INTEGER NOT NULL,
    horse_no INTEGER NOT NULL,
    horse_id TEXT,
    horse_name TEXT,
    jockey_id TEXT,
    jockey_name TEXT,
    trainer_id TEXT,
    trainer_name TEXT,
    weight INTEGER,
    draw INTEGER,
    rating INTEGER,
    rating_change INTEGER,
    body_weight INTEGER,
    last6 TEXT,
    gear TEXT,
    age INTEGER,
    sex TEXT,
    days_since_last INTEGER,
    jockey_allowance INTEGER,        -- apprentice claim in lb (e.g. 7 for "(-7)"), 0 if none
    scratched INTEGER DEFAULT 0,
    scraped_at TEXT,
    PRIMARY KEY (date, venue, race_no, horse_no)
);

CREATE TABLE IF NOT EXISTS odds_snapshots (
    date TEXT NOT NULL,
    venue TEXT NOT NULL,
    race_no INTEGER NOT NULL,
    pool TEXT NOT NULL,
    combination TEXT NOT NULL,
    odds REAL,
    snapshot_at TEXT NOT NULL,
    PRIMARY KEY (date, venue, race_no, pool, combination, snapshot_at)
);

CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,              -- post_analysis / pre_bet / walkforward / backtest
    date TEXT,
    venue TEXT,
    created_at TEXT,
    notes TEXT                       -- JSON
);

CREATE TABLE IF NOT EXISTS predictions (
    run_id TEXT NOT NULL,
    race_id TEXT NOT NULL,
    horse_no INTEGER NOT NULL,
    p_win REAL,
    p_place REAL,
    model_version TEXT,
    created_at TEXT,
    PRIMARY KEY (run_id, race_id, horse_no)
);

CREATE TABLE IF NOT EXISTS paper_tickets (
    ticket_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT,
    date TEXT NOT NULL,
    venue TEXT NOT NULL,
    strategy TEXT NOT NULL,          -- PLA_6x63 / QPL_2x3 / ...
    ticket_no INTEGER,
    legs TEXT NOT NULL,              -- JSON [{race_no, selections:[...]}]
    unit_stake REAL,
    n_combos INTEGER,
    total_stake REAL,
    status TEXT DEFAULT 'open',      -- open / settled / void
    payout REAL,
    profit REAL,
    created_at TEXT,
    settled_at TEXT,
    detail TEXT                      -- JSON
);

CREATE TABLE IF NOT EXISTS bankroll_ledger (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT,
    ticket_id INTEGER,
    amount REAL,
    balance REAL,
    note TEXT
);

CREATE TABLE IF NOT EXISTS backtest_results (
    bt_id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy TEXT,
    params TEXT,                     -- JSON
    period_start TEXT,
    period_end TEXT,
    n_bets INTEGER,
    n_hits INTEGER,
    hit_rate REAL,
    stake REAL,
    returns REAL,
    roi REAL,
    max_drawdown REAL,
    created_at TEXT
);
"""


def connect(path=None) -> sqlite3.Connection:
    path = Path(path) if path is not None else db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=60)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    init_db(conn)
    return conn


MIGRATIONS = [
    ("racecard", "jockey_allowance", "INTEGER"),
    ("paper_tickets", "bot_id", "TEXT"),        # which robot placed it (see hkjc/bots.py); NULL = legacy
    ("paper_tickets", "reason", "TEXT"),        # 中文落注原因
    ("bankroll_ledger", "bot_id", "TEXT"),      # each robot has its own bankroll
]

EXTRA_SCHEMA = """
CREATE TABLE IF NOT EXISTS horse_scores (
    run_id TEXT NOT NULL,
    race_id TEXT NOT NULL,
    horse_no INTEGER NOT NULL,
    score REAL,                      -- 綜合評分 0-100
    rank_in_race INTEGER,
    sub_scores TEXT,                 -- JSON {form, jockey, trainer, combo, draw, fit, class, market}: 0-100
    reasons TEXT,                    -- JSON list of short 中文 reasons
    created_at TEXT,
    PRIMARY KEY (run_id, race_id, horse_no)
);
CREATE TABLE IF NOT EXISTS bot_decisions (
    run_id TEXT NOT NULL,
    bot_id TEXT NOT NULL,
    date TEXT NOT NULL,
    venue TEXT NOT NULL,
    race_no INTEGER NOT NULL,        -- 0 = meeting-level (e.g. an all-up across races)
    action TEXT NOT NULL,            -- bet / skip
    structure TEXT,                  -- e.g. 'PLA 單注', 'QIN 膽1x腳3', 'PLA 6x63'
    selections TEXT,                 -- JSON
    stake REAL,
    reason TEXT,                     -- 中文原因
    ticket_id INTEGER,
    created_at TEXT,
    PRIMARY KEY (run_id, bot_id, race_no, structure)
);
"""


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.executescript(EXTRA_SCHEMA)
    for table, col, typ in MIGRATIONS:
        cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
        if col not in cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typ}")
    conn.commit()


def upsert(conn: sqlite3.Connection, table: str, rows) -> int:
    """INSERT OR REPLACE a list of dicts (all with the same keys)."""
    rows = list(rows)
    if not rows:
        return 0
    cols = list(rows[0].keys())
    sql = f"INSERT OR REPLACE INTO {table} ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})"
    conn.executemany(sql, [tuple(r.get(c) for c in cols) for r in rows])
    conn.commit()
    return len(rows)
