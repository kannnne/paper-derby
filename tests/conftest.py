"""Shared test safety net.

cli.cmd_pre / cmd_post end with cli._publish -> publish.publish(conn), whose defaults are
the REAL site/ (its own git repo) and data/publish.json.  A test that exercises cmd_pre /
cmd_post would otherwise export its synthetic DB into site/data and git-commit it there.
Every test therefore gets a throw-away site dir, publish config and report dir."""
import pytest


@pytest.fixture(autouse=True)
def _isolate_site_and_reports(tmp_path_factory, monkeypatch):
    from hkjc import cli, publish

    base = tmp_path_factory.mktemp("isolated")
    monkeypatch.setattr(publish, "SITE_DIR", base / "site")
    monkeypatch.setattr(publish, "PUBLISH_CONFIG", base / "publish.json")
    monkeypatch.setattr(cli, "REPORT_DIR", base / "reports")
    yield base
