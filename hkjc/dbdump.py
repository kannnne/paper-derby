"""Text dump / restore of the SQLite DB so it can live in git (GitHub Actions cloud mode).

    python -m hkjc.dbdump dump data/hkjc.db data/db     # one <table>.sql per table
    python -m hkjc.dbdump load data/db data/hkjc.db     # rebuild a fresh DB from the dump

Rows are written as INSERT statements ordered by primary key so git diffs stay small.
Floats use repr() (exact round-trip). Volatile columns in VOLATILE are not dumped
(e.g. meetings.updated_at, a cache timestamp) so idle runs don't create commits.
"""
import sqlite3
import sys
from pathlib import Path

from . import db as dbmod

VOLATILE = {"meetings": {"updated_at"}}
SKIP_TABLES = {"sqlite_sequence"}


def _lit(v) -> str:
    if v is None:
        return "NULL"
    if isinstance(v, bool):
        return str(int(v))
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if v != v:
            return "NULL"
        return format(v, ".15g")  # 15 sig. digits: identical round-trip on every platform
    if isinstance(v, (bytes, bytearray)):
        return "X'" + bytes(v).hex() + "'"
    s = "'" + str(v).replace("'", "''") + "'"
    # keep one statement per line: newlines / CRs become char() concatenations
    return s.replace("\r", "' || char(13) || '").replace("\n", "' || char(10) || '")


def _tables(conn):
    return [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name") if r[0] not in SKIP_TABLES]


def _cols(conn, table):
    info = conn.execute(f"PRAGMA table_info({table})").fetchall()
    cols = [r[1] for r in info]
    pk = [r[1] for r in sorted(info, key=lambda r: r[5]) if r[5] > 0]
    return cols, pk


def dump(db_path, out_dir) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    written = []
    tables = _tables(conn)
    for t in tables:
        cols, pk = _cols(conn, t)
        keep = sorted(c for c in cols if c not in VOLATILE.get(t, set()))  # stable across schema histories
        order = ", ".join(pk) if pk else "rowid"
        path = out / f"{t}.sql"
        tmp = path.with_suffix(".sql.tmp")
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            head = f"INSERT INTO {t} ({','.join(keep)}) VALUES ("
            for row in conn.execute(f"SELECT {','.join(keep)} FROM {t} ORDER BY {order}"):
                f.write(head + ",".join(_lit(v) for v in row) + ");\n")
        tmp.replace(path)
        written.append(path)
    for stale in out.glob("*.sql"):
        if stale.stem not in tables:
            stale.unlink()
    conn.close()
    return written


def load(in_dir, db_path) -> int:
    in_dir, db_path = Path(in_dir), Path(db_path)
    for suffix in ("", "-wal", "-shm"):
        p = Path(str(db_path) + suffix)
        if p.exists():
            p.unlink()
    conn = dbmod.connect(db_path)          # creates the current schema (+ migrations)
    n = 0
    conn.execute("BEGIN")
    for path in sorted(in_dir.glob("*.sql")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    conn.execute(line)
                    n += 1
    conn.commit()
    conn.close()
    return n


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 3 or argv[0] not in ("dump", "load"):
        print(__doc__)
        return 2
    if argv[0] == "dump":
        print(f"dumped {len(dump(argv[1], argv[2]))} tables")
    else:
        print(f"loaded {load(argv[1], argv[2])} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
