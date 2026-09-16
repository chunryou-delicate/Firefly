"""M5b: static checks on the flysim-live cockpit (docs/m5b-brief.md).

No browser automation. These tests check that the single HTML file parses, pulls in
no external code, and implements every message type the contract (docs/m5-protocol.md)
defines — the message lists are read out of the contract itself, so a contract change
that the cockpit has not followed fails here.
"""
from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT / "flysim-live.html"
VIEWER = ROOT / "flysim-viewer.html"
PROTOCOL = ROOT / "docs" / "m5-protocol.md"

ALLOWED_LINK_HOSTS = ("https://fonts.googleapis.com", "https://fonts.gstatic.com")


@pytest.fixture(scope="module")
def html() -> str:
    return LIVE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def script(html) -> str:
    """The inline script body (everything the cockpit runs)."""
    m = re.search(r"<script>\n(.*)\n</script>", html, re.S)
    assert m, "inline <script> block not found"
    return m.group(1)


class _Collector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags: list[tuple[str, dict]] = []

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))


@pytest.fixture(scope="module")
def tags(html):
    c = _Collector()
    c.feed(html)
    return c.tags


def _contract_types(section: str) -> set[str]:
    """Message type names from one table of docs/m5-protocol.md."""
    text = PROTOCOL.read_text(encoding="utf-8")
    start = text.index(section)
    rest = text[start + len(section):]
    end = rest.find("\n## ")
    body = rest[:end if end > 0 else len(rest)]
    out: set[str] = set()
    for line in body.splitlines():
        if not line.startswith("|") or line.startswith("|---"):
            continue
        first = line.split("|")[1]
        names = re.findall(r"`([a-z_]+)`", first)
        out.update(n for n in names if n not in {"type"})
    return out


# ---------------------------------------------------------------- file shape
def test_file_exists_and_parses(html, tags):
    assert html.lstrip().startswith("<!DOCTYPE html>")
    assert tags, "no tags parsed"
    assert "<title>" in html


def test_no_external_scripts(tags):
    for tag, attrs in tags:
        if tag == "script":
            assert "src" not in attrs, f"external script not allowed: {attrs.get('src')}"


def test_only_font_links_are_external(tags):
    for tag, attrs in tags:
        if tag in ("link", "img", "iframe", "object", "embed"):
            url = attrs.get("href") or attrs.get("src") or ""
            if url.startswith(("http://", "https://", "//")):
                assert url.startswith(ALLOWED_LINK_HOSTS), f"external resource not allowed: {url}"


def test_no_inline_style_or_script_files_needed(html):
    assert "<style>" in html, "stylesheet must be inline (single file)"
    assert html.count("<script") == 1, "exactly one inline script expected"


def test_viewer_is_a_separate_file_and_is_credited(html):
    """The cockpit is a second file; the result player stays. The brief allows copying the
    field-drawing approach from the viewer as long as the source is credited in a comment."""
    assert VIEWER.exists(), "the result player must still be there"
    title = re.search(r"<title>(.*?)</title>", html).group(1)
    assert "flysim-live" in title and "재생기" not in title
    assert re.search(r"//[^\n]*flysim-viewer\.html", html), \
        "missing attribution comment for the copied field renderer"


# ------------------------------------------------------- contract: incoming
def test_server_message_types_match_contract(script):
    contract = _contract_types("## 서버 → 클라이언트")
    assert contract, "could not read the server->client table from the contract"
    declared = set(re.findall(r'"([a-z_]+)"', re.search(
        r"const SERVER_MESSAGE_TYPES = \[(.*?)\];", script, re.S).group(1)))
    assert declared == contract, f"declared {declared} != contract {contract}"


@pytest.mark.parametrize("msg_type", sorted(_contract_types("## 서버 → 클라이언트")))
def test_every_server_message_has_a_handler(script, msg_type):
    """Each incoming type must be a branch of the dispatch table, bound to a function."""
    table = re.search(r"const SERVER_HANDLERS = \{(.*?)\};", script, re.S)
    assert table, "SERVER_HANDLERS dispatch table not found"
    m = re.search(rf"\b{msg_type}:\s*(\w+),", table.group(1))
    assert m, f"no handler entry for {msg_type!r}"
    assert re.search(rf"function {m.group(1)}\(", script), f"handler {m.group(1)} not defined"


# ------------------------------------------------------- contract: outgoing
def test_client_message_types_match_contract(script):
    contract = _contract_types("## 클라이언트 → 서버")
    assert contract, "could not read the client->server table from the contract"
    declared = set(re.findall(r'"([a-z_]+)"', re.search(
        r"const CLIENT_MESSAGE_TYPES = \[(.*?)\];", script, re.S).group(1)))
    assert declared == contract, f"declared {declared} != contract {contract}"


@pytest.mark.parametrize("msg_type", sorted(_contract_types("## 클라이언트 → 서버")))
def test_every_client_message_is_actually_sent(script, msg_type):
    """Each outgoing type must have a real send() call site, not just a declaration."""
    assert re.search(rf'send\(\s*"{msg_type}"', script) or \
           re.search(rf'send\(\s*S\.status === "paused" \? "resume" : "{msg_type}"', script) or \
           re.search(rf'\? "{msg_type}" : ', script), f"no send() call for {msg_type!r}"


def test_send_validates_against_the_contract_list(script):
    assert "CLIENT_MESSAGE_TYPES.includes(type)" in script
    assert "req_id" in script and "S.reqId++" in script, "every client message needs a req_id"


def test_params_object_keys_match_contract(script):
    text = PROTOCOL.read_text(encoding="utf-8")
    line = text[text.index("`params` 객체:"):].split("\n")[0]
    block = line[line.index("{"):line.rindex("}") + 1]
    contract = set(re.findall(r"(\w+):", block))
    declared = set(re.findall(r'"(\w+)"', re.search(
        r"const PARAM_KEYS = \[(.*?)\];", script, re.S).group(1)))
    assert declared == contract, f"declared {declared} != contract {contract}"


def test_binary_audio_path_exists(script):
    assert "function sendBinary" in script
    assert "Float32Array" in script, "audio chunks are float32 PCM"
    assert 'send("audio_start"' in script and 'send("audio_stop"' in script


# ------------------------------------------------------------- behaviour bits
def test_mock_mode_branch_exists(html, script):
    assert 'q.get("mock") === "1"' in script, "?mock=1 branch missing"
    assert "class MockServer" in script
    assert "mockBadge" in html, "mock mode must be labelled on screen"
    assert "합성" in html, "mock data must be declared synthetic"


def test_warning_codes_from_contract_are_handled(script):
    text = PROTOCOL.read_text(encoding="utf-8")
    row = [l for l in text.splitlines() if l.startswith("| `warning`")][0]
    codes = set(re.findall(r'`"(\w+)"`', row))
    declared = set(re.findall(r'"(\w+)"', re.search(
        r"const WARNING_CODES = \[(.*?)\];", script, re.S).group(1)))
    assert declared == codes, f"declared {declared} != contract {codes}"
    for code in codes:
        assert code in re.search(r"const WARNING_TEXT = \{(.*?)\};", script, re.S).group(1), \
            f"no user-facing text for warning {code!r}"
    assert '"_cleared"' in script, "warning clearing must be handled"


def test_normalisation_modes_and_default(html, script):
    modes = set(re.findall(r'data-norm="(\w+)"', html))
    assert modes == {"region", "global", "log", "fixed"}
    assert re.search(r'data-norm="region"[^>]*class="on"', html), "default must be per-region"
    assert 'norm: "region"' in script, "initial state must be per-region"


def test_client_side_normalisation_only(script):
    """The server sends raw Hz; the cockpit must not expect normalised rates."""
    assert "정규화는 전부" in script or "정규화는 전부" in LIVE.read_text(encoding="utf-8")
    assert "rowMax" in script, "per-region maxima are computed client-side"


def test_no_optimistic_param_updates(script):
    """S.params may only be written from hello and from a successful ack."""
    writes = re.findall(r"Object\.assign\(S\.params", script)
    assert len(writes) == 1, "S.params should be merged in exactly one place (the ack handler)"
    ack = re.search(r"function onAck\(msg\)\{(.*?)\n\}", script, re.S).group(1)
    assert "Object.assign(S.params, sent)" in ack
    assert "if(msg.ok)" in ack, "params must only be applied when the server accepted them"


def test_reconnect_interval(script):
    assert "const RECONNECT_MS = 2000;" in script
    assert "setTimeout(connect, RECONNECT_MS)" in script


def test_snapshot_is_marked_not_for_submission(html):
    assert "결과 제출용이 아니다" in html


def test_control_log_notice_is_shown(html):
    assert "control-log.jsonl" in html or "기록된다" in html


def test_responsive_rules_present(html):
    assert "@media (max-width:900px)" in html
    assert "@media (max-width:420px)" in html


def test_panel_warning_class_does_not_collide_with_field_overlay(html):
    """`.warn` is the absolutely positioned field overlay; panel notes use .note.caution.
    Using `class="note warn"` would rip the note out of the panel flow."""
    assert 'class="note warn"' not in html
    assert ".note.caution{" in html
