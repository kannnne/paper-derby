"""GitHub Actions runner loop: GitHub's cron fires only every few hours, so one job keeps
running `python -m hkjc tick` every few minutes for ~5.5 h, commits data, publishes the
site to the gh-pages branch, and finally dispatches the next run (self-chaining 24/7).

    python scripts/cloud_loop.py loop --minutes 330 --interval 5
    python scripts/cloud_loop.py once "scores"        # one command, then commit + publish

Assumes: cwd = repo root, data/hkjc.db restored from data/db, git identity configured.
"""
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone

HKT = timezone(timedelta(hours=8))
PY = sys.executable
CODE_PATHS = ("hkjc", "scripts", ".github", "site/app.js", "site/style.css", "site/index.html", "requirements.txt")


def log(msg):
    print(f"[loop {datetime.now(HKT):%Y-%m-%d %H:%M:%S} HKT] {msg}", flush=True)


def sh(cmd, check=False):
    r = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    if r.stdout.strip():
        print(r.stdout.rstrip()[-4000:], flush=True)
    if r.returncode and r.stderr.strip():
        print(r.stderr.rstrip()[-4000:], flush=True)
    if check and r.returncode:
        raise RuntimeError(f"failed: {cmd}")
    return r


def code_rev():
    return sh("git rev-parse HEAD:hkjc HEAD:scripts HEAD:.github HEAD:site/app.js HEAD:site/index.html").stdout


def publish_pages():
    """Force-push the committed site/ tree as a single orphan commit to gh-pages."""
    tree = sh("git rev-parse HEAD:site").stdout.strip()
    if not tree:
        return
    commit = sh(f'git commit-tree {tree} -m "site {datetime.now(HKT):%Y-%m-%d %H:%M} HKT"').stdout.strip()
    r = sh(f"git push -q -f origin {commit}:refs/heads/gh-pages")
    log("pages published" if r.returncode == 0 else "pages push failed")


def commit_and_push(msg):
    sh(f"{PY} -m hkjc.dbdump dump data/hkjc.db data/db")
    sh("git add -A data/db reports site data/bot_backtest.json")
    if sh("git diff --cached --quiet").returncode == 0:
        return False
    sh(f'git commit -q -m "{msg}"')
    for _ in range(3):
        sh("git pull -q --rebase")
        if sh("git push -q").returncode == 0:
            publish_pages()
            return True
        time.sleep(5)
    log("push failed 3x")
    return False


def run_tick(cmd="tick"):
    r = sh(f"{PY} -m hkjc {cmd}")
    if r.returncode:
        log(f"`hkjc {cmd}` exited {r.returncode}")


def loop(minutes, interval):
    end = time.time() + minutes * 60
    rev = code_rev()
    restart = False
    while time.time() < end:
        sh("git pull -q --rebase")
        if code_rev() != rev:
            log("code changed on main -> restarting to load it")
            restart = True
            break
        run_tick()
        commit_and_push(f"bot: {datetime.now(HKT):%Y-%m-%d %H:%M} HKT")
        time.sleep(max(30, interval * 60))
    log("loop finished" + (" (restart)" if restart else ""))


def dispatch_next():
    r = sh("gh workflow run race-bot.yml -f command=loop")
    log("next run dispatched" if r.returncode == 0 else "dispatch failed (the hourly schedule will restart it)")


def main(argv):
    if not argv or argv[0] not in ("loop", "once"):
        print(__doc__)
        return 2
    if argv[0] == "once":
        run_tick(argv[1] if len(argv) > 1 else "tick")
        commit_and_push(f"bot: {datetime.now(HKT):%Y-%m-%d %H:%M} HKT ({argv[1] if len(argv) > 1 else 'tick'})")
        publish_pages()
        return 0
    minutes = int(argv[argv.index("--minutes") + 1]) if "--minutes" in argv else 330
    interval = float(argv[argv.index("--interval") + 1]) if "--interval" in argv else 5
    try:
        loop(minutes, interval)
    finally:
        dispatch_next()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
