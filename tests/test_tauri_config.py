"""Tests for the Tauri shell's offline security posture (SPEC §10, §12, §17).

The desktop shell must be strictly offline: its Content-Security-Policy may not
allow any remote origin (only the app itself and the localhost sidecar), and its
capability set must stay minimal. These read the *shipped* config so a regression
that opens a network hole or widens the allowlist fails CI — independent of the
Rust toolchain.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

SRC_TAURI = Path(__file__).resolve().parent.parent / "app" / "src-tauri"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_csp_forbids_remote_origins():
    """The CSP confines the webview to itself + the localhost sidecar — nothing
    routable to the internet (SPEC §10 strict offline posture)."""
    conf = _load_json(SRC_TAURI / "tauri.conf.json")
    csp = conf["app"]["security"]["csp"]
    assert csp, "no CSP configured — the offline posture is unenforced"
    # Every host source must be a loopback address; nothing routable.
    hosts = re.findall(r"(?:https?|wss?)://[^\s;'\"]+", csp)
    assert hosts, "CSP names no connect hosts at all (expected the localhost sidecar)"
    for host in hosts:
        assert re.search(r"(?:127\.0\.0\.1|localhost)", host), f"non-local origin in CSP: {host}"
    assert "http://*" not in csp and "https://*" not in csp, "wildcard remote origin in CSP"


def test_capabilities_are_minimal_and_offline():
    """No shell/exec or internet-HTTP capability; only the webview core plus local
    file dialogs/fs for study.json (SPEC §10 minimal allowlist)."""
    cap = _load_json(SRC_TAURI / "capabilities" / "default.json")
    perm_ids = [
        p if isinstance(p, str) else p.get("identifier", "") for p in cap["permissions"]
    ]
    assert perm_ids, "capability set is empty"
    for pid in perm_ids:
        namespace = pid.split(":")[0]
        assert namespace in {"core", "dialog", "fs"}, f"capability outside minimal set: {pid}"
    joined = " ".join(perm_ids)
    assert "shell" not in joined and "http:" not in joined, "network/exec capability present"
