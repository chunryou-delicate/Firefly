"""M6c: static checks on the 3D neuron viewer (docs/m6c-brief.md).

No browser automation. The checks read `docs/m6-3d-contract.md` and
`flysim-viewer.html` directly, so a contract change or a viewer wording change
that the 3D viewer has not followed fails here rather than silently drifting.
"""
from __future__ import annotations

import json
import re
from html.parser import HTMLParser
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VIEW3D = ROOT / "flysim-3d.html"
VIEWER = ROOT / "flysim-viewer.html"
COCKPIT = ROOT / "flysim-live.html"
CONTRACT = ROOT / "docs" / "m6-3d-contract.md"

ALLOWED_LINK_HOSTS = ("https://fonts.googleapis.com", "https://fonts.gstatic.com")
# 3D 라이브러리를 들여오지 않는다는 것을 이름으로 못박아 둔다
FORBIDDEN_LIBS = ("three.min.js", "three.module", "babylon", "regl", "twgl",
                  "cdn.jsdelivr", "unpkg.com", "cdnjs.cloudflare")


@pytest.fixture(scope="module")
def html() -> str:
    return VIEW3D.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def script(html) -> str:
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


def _contract_blocks() -> list[dict]:
    """The json examples in the contract use placeholders ([x,y,z], "a | null").
    Substitute those so the shape can be checked; the keys are what matters here."""
    text = CONTRACT.read_text(encoding="utf-8")
    out = []
    for raw in re.findall(r"```json\n(\{.*?\n\})\n```", text, re.S):
        cleaned = raw.replace("[x,y,z] | null", "null").replace("[x,y,z]", "[0,0,0]")
        out.append(json.loads(cleaned))
    assert out, "no json examples found in the contract"
    return out


@pytest.fixture(scope="module")
def contract_cloud() -> dict:
    for d in _contract_blocks():
        if "arrays" in d:
            return d
    pytest.fail("could not find the point-cloud json example in the contract")


@pytest.fixture(scope="module")
def contract_skel() -> dict:
    for d in _contract_blocks():
        if "n_segments" in d:
            return d
    pytest.fail("could not find the skeleton bundle json example in the contract")


# ---------------------------------------------------------------- file shape
def test_file_parses_and_is_self_contained(html, tags):
    assert html.lstrip().startswith("<!DOCTYPE html>")
    assert "<title>" in html and "<style>" in html
    assert html.count("<script") == 1, "exactly one inline script expected"
    assert tags


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


def test_no_3d_library_is_pulled_in(tags, script):
    """WebGL2 is used directly; three.js and friends are forbidden by the brief."""
    for tag, attrs in tags:
        url = (attrs.get("href") or attrs.get("src") or "").lower()
        for bad in FORBIDDEN_LIBS:
            assert bad not in url, f"{bad} must not be loaded"
    for bad in FORBIDDEN_LIBS:
        assert bad not in script.lower(), f"{bad} referenced in the script"
    assert "THREE." not in script, "no three.js global"


def test_uses_webgl2_directly(script):
    assert 'getContext("webgl2"' in script
    assert script.count("#version 300 es") >= 2, "GLSL ES 3.00 shaders expected"
    for fn in ("createShader", "linkProgram", "createVertexArray", "drawArrays"):
        assert fn in script, f"raw WebGL call {fn} missing"
    assert "gl.POINTS" in script and "gl.LINES" in script


def test_webgl2_unsupported_message(html, script):
    assert "WebGL2" in html, "there must be a message for browsers without WebGL2"
    assert 'classList.add("blocked")' in script


def test_other_html_surfaces_still_exist(html):
    """The 2D player and the cockpit keep their roles; this is a third file."""
    assert VIEWER.exists() and COCKPIT.exists()
    title = re.search(r"<title>(.*?)</title>", html).group(1)
    assert "flysim-3d" in title


# ------------------------------------------------------------- the contract
def test_voxel_size_matches_contract(script):
    text = CONTRACT.read_text(encoding="utf-8")
    assert "1 복셀 = **8 nm**" in text, "contract changed the voxel size"
    assert "const VOXEL_NM = 8;" in script
    assert "UM_PER_VOXEL = VOXEL_NM / 1000" in script


def test_point_cloud_array_names_match_contract(script, contract_cloud):
    names = set(contract_cloud["arrays"])
    declared = set(re.findall(r'"(\w+)"', re.search(
        r"const CLOUD_ARRAYS = \[(.*?)\];", script, re.S).group(1)))
    assert declared == names, f"declared {declared} != contract {names}"


@pytest.mark.parametrize("name", ["pos", "set", "src"])
def test_each_contract_array_is_read(script, contract_cloud, name):
    spec = contract_cloud["arrays"][name]
    assert spec["dtype"] in re.search(r"const DTYPES = \{(.*?)\};", script, re.S).group(1), \
        f"dtype {spec['dtype']} not handled"
    assert re.search(rf'readArray\(buf, a\.{name}, "{name}"\)', script), \
        f"arrays.{name} is never read"


def test_offsets_are_bytes_and_lengths_are_elements(script, contract_cloud):
    """The contract's own numbers only line up under that reading; assert the code says so."""
    a = contract_cloud["arrays"]
    assert a["pos"]["offset"] + a["pos"]["length"] * 4 == a["set"]["offset"]
    assert a["set"]["offset"] + a["set"]["length"] * 1 == a["src"]["offset"]
    body = re.search(r"function readArray\((.*?)\n\}", script, re.S).group(1)
    assert "BYTES_PER_ELEMENT" in body, "length must be scaled by the element size"
    assert "spec.offset" in body and "spec.length" in body
    assert "buf.byteLength" in body, "the reader must bounds-check against the file"


def test_missing_coordinates_are_dropped_not_zeroed(script):
    """Contract: src = 2 means NaN; they must not be collapsed onto the origin."""
    text = CONTRACT.read_text(encoding="utf-8")
    assert "NaN" in text and "0으로 채워서" in text
    assert "const SRC_NONE = 2;" in script
    assert "Number.isFinite" in script, "non-finite coordinates must be filtered"
    assert "neuronToDraw" in script, "a neuron index -> draw index map is needed after filtering"


def test_skeleton_bundle_fields_are_used(script, contract_skel):
    for key in ("n_segments", "neurons", "missing"):
        assert key in script, f"skeleton bundle field {key} unused"
    for key in ("seg_offset", "seg_count", "body_id"):
        assert key in contract_skel["neurons"][0]
        assert key in script, f"per-neuron field {key} unused"
    assert "gl.LINES" in script, "segments are drawn as GL_LINES per the contract"


def test_http_endpoints_match_contract(script):
    text = CONTRACT.read_text(encoding="utf-8")
    rows = [l for l in text.splitlines() if l.startswith("| `/")]
    paths = set()
    for r in rows:
        paths.update(re.findall(r"`/([^`]+)`", r.split("|")[1]))
    wanted = {p.replace("<name>", "") for p in paths}
    body = re.search(r"const ENDPOINTS = \{(.*?)\};", script, re.S).group(1)
    for w in wanted:
        stem = w.replace(".json", "").replace(".bin", "").strip("/")
        assert stem.split("/")[0] in body, f"endpoint {w} not implemented"
    assert "neurons-3d.json" in body and "neurons-3d.bin" in body
    assert "skel/index.json" in body


def test_no_unit_conversion_or_recentring_on_load(script):
    """Contract forbids converting units or re-centring inside the files; display only."""
    parse = re.search(r"function parseCloud\((.*?)\n\}", script, re.S).group(1)
    assert "UM_PER_VOXEL" not in parse, "parsing must not convert voxels to microns"
    assert "UM_PER_VOXEL" in script, "but the display should report microns"


# --------------------------------------------------------------- behaviour
def test_mock_mode_branch_and_red_badge(html, script):
    assert 'q.get("mock") === "1"' in script, "?mock=1 branch missing"
    assert "function loadMock" in script
    assert "mockBadge" in html
    assert re.search(r'id="mockBadge"[^>]*class="badge synthetic"|class="badge synthetic"[^>]*id="mockBadge"', html), \
        "the mock badge must use the red synthetic style"
    assert "합성" in html, "synthetic data must be declared on screen"


def test_mock_builds_contract_shaped_buffers(script):
    """The mock must go through the same parser as real files, not a shortcut."""
    assert "setCloud(parseCloud(meta, buf), true)" in script
    assert "parseSkel(" in script and "addSkel(parseSkel(" in script
    assert "new ArrayBuffer(" in script, "the mock serialises to the contract's binary layout"


def test_failure_wording_matches_the_2d_viewer(script):
    """The brief says to keep the existing viewer's failure detection wording."""
    viewer = VIEWER.read_text(encoding="utf-8")
    silent = re.search(r'w\.textContent = "(발화 뉴런 없음[^"]*)"', viewer).group(1)
    runaway = re.search(r'w\.textContent = "(전체의 90%[^"]*)"', viewer).group(1)
    assert f'WARN_SILENT  = "{silent}"' in script or f'WARN_SILENT = "{silent}"' in script
    assert f'WARN_RUNAWAY = "{runaway}"' in script
    assert "0.90" in script, "the 90 % threshold must be applied"


def test_positions_uploaded_once_activity_updated_per_frame(script):
    """Brief: upload the 165k positions once and only refresh colour each frame."""
    assert "gl.bufferData(gl.ARRAY_BUFFER, c.pos, gl.STATIC_DRAW)" in script
    assert "gl.bufferSubData(gl.ARRAY_BUFFER, 0, act)" in script
    assert "gl.DYNAMIC_DRAW" in script
    render = re.search(r"function render\(ts\)\{(.*?)\n\}", script, re.S).group(1)
    assert "bufferData" not in render, "positions must not be re-uploaded every frame"


def test_one_buffer_per_skeleton_set(script):
    add = re.search(r"function addSkel\((.*?)\n\}", script, re.S).group(1)
    assert "gl.bufferData(gl.ARRAY_BUFFER, bundle.segs, gl.STATIC_DRAW)" in add


def test_camera_controls_present(script, html):
    assert "pointerdown" in script and "wheel" in script
    assert "shiftKey" in script, "Shift+drag panning"
    for p in ("front", "side", "top"):
        assert f'data-preset="{p}"' in html
    assert "spinBtn" in script, "auto-rotate toggle"


def test_reduced_motion_disables_auto_rotation(script, html):
    assert "prefers-reduced-motion" in script
    assert "S.reduceMotion" in script
    assert "S.spin && !S.reduceMotion" in script


def test_run_json_is_accepted_by_drop_and_picker(script):
    assert "dataTransfer.files" in script
    assert "readRunFile" in script and "FileReader" in script
    assert "frames.rates" in script or "fr.rates" in script
    assert "spikes?.neuron_idx" in script or "spikes.neuron_idx" in script


def test_run_json_contract_is_not_modified(script):
    """Activity still comes from the existing run.json; the 3D viewer only reads it."""
    fn = re.search(r"function loadRun\((.*?)\n\}", script, re.S).group(1)
    for key in ("frames", "regions", "spikes"):
        assert key in fn
    assert "neuronToDraw" in fn, "neuron indices map into the point cloud order"


def test_missing_skeleton_files_are_a_notice_not_an_error(script):
    assert "skelNote" in script
    body = re.search(r"async function loadFromServer\((.*?)\n\}", script, re.S).group(1)
    assert "catch" in body, "a missing bundle must not abort loading"


# ═══════════════════════════════════════════════════════════════════
#  M6d — 실시간 모드, 뉴런 집기, 하류·골격 강조
# ═══════════════════════════════════════════════════════════════════
PROTOCOL = ROOT / "docs" / "m5-protocol.md"


def _protocol_types(section: str) -> set[str]:
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
        out.update(n for n in re.findall(r"`([a-z_]+)`", line.split("|")[1]) if n != "type")
    return out


# ---------------------------------------------------- contract v1.3: body
def test_contract_v13_body_array_is_optional(script, contract_cloud):
    """v1.3 added `body`; files written before it have no such array and must still load."""
    text = CONTRACT.read_text(encoding="utf-8")
    assert "**계약 버전 v1.3**" in text, "this viewer implements 3D contract v1.3"
    row = [l for l in text.splitlines() if l.startswith("| `body`")]
    assert row, "the contract must describe the body array"
    assert "uint32" in row[0]
    required = set(re.findall(r'"(\w+)"', re.search(
        r"const CLOUD_ARRAYS = \[(.*?)\];", script, re.S).group(1)))
    optional = set(re.findall(r'"(\w+)"', re.search(
        r"const CLOUD_ARRAYS_OPTIONAL = \[(.*?)\];", script, re.S).group(1)))
    assert "body" in optional and "body" not in required, \
        "body must be optional, or old point clouds stop loading"
    assert required == set(contract_cloud["arrays"]) - {"body"}


def test_body_array_read_and_absence_tolerated(script):
    parse = re.search(r"function parseCloud\((.*?)\n\}", script, re.S).group(1)
    assert "if(a.body)" in parse, "body is read only when the file has it"
    assert 'readArray(buf, a.body, "body")' in parse
    sel = re.search(r"function selectNeuron\((.*?)\n\}", script, re.S).group(1)
    assert "if(c.body)" in sel, "the panel must fall back when body is absent"
    assert "계약 v1.3 이전" in sel, "say why the bodyId is missing instead of showing nothing"


def test_uint32_is_a_known_dtype(script):
    assert "uint32" in re.search(r"const DTYPES = \{(.*?)\};", script, re.S).group(1)


# ------------------------------------------------- protocol v1.2: live mode
def test_live_message_types_are_a_subset_of_the_protocol(script):
    server = _protocol_types("## 서버 → 클라이언트")
    client = _protocol_types("## 클라이언트 → 서버")
    assert server and client, "could not read the protocol tables"
    declared_s = set(re.findall(r'"(\w+)"', re.search(
        r"const LIVE_SERVER_TYPES = \[(.*?)\];", script, re.S).group(1)))
    declared_c = set(re.findall(r'"(\w+)"', re.search(
        r"const LIVE_CLIENT_TYPES = \[(.*?)\];", script, re.S).group(1)))
    assert declared_s <= server, f"invented server types: {declared_s - server}"
    assert declared_c <= client, f"invented client types: {declared_c - client}"
    # this screen consumes every server message the protocol defines
    assert declared_s == server, f"unhandled server messages: {server - declared_s}"


def test_no_parameter_control_from_the_3d_screen(script):
    """The brief is explicit: fine control stays in the 2D cockpit, or both screens
    grow the same feature and every fix has to happen twice."""
    declared_c = set(re.findall(r'"(\w+)"', re.search(
        r"const LIVE_CLIENT_TYPES = \[(.*?)\];", script, re.S).group(1)))
    assert "set_params" not in declared_c
    assert 'sendLive("set_params"' not in script
    assert "sendLive" in script and "LIVE_CLIENT_TYPES.includes(type)" in script


@pytest.mark.parametrize("msg_type", ["stimulus", "pause", "resume", "step", "reset",
                                      "get_hops", "snapshot"])
def test_each_live_client_message_is_sent(script, msg_type):
    assert re.search(rf'sendLive\(\s*"{msg_type}"', script) or \
           re.search(rf'\?\s*"resume"\s*:\s*"{msg_type}"', script) or \
           re.search(rf'sendLive\(S\.liveStatus === "paused" \? "{msg_type}"', script), \
        f"no sendLive() call for {msg_type!r}"


@pytest.mark.parametrize("msg_type", ["hello", "frame", "params", "ack", "hops",
                                      "snapshot_done", "warning", "error"])
def test_each_live_server_message_has_a_handler(script, msg_type):
    table = re.search(r"const LIVE_HANDLERS = \{(.*?)\};", script, re.S).group(1)
    m = re.search(rf"\b{msg_type}:\s*(\w+),", table)
    assert m, f"no handler for {msg_type!r}"
    assert re.search(rf"function {m.group(1)}\(", script), f"{m.group(1)} is not defined"


def test_ws_query_must_be_a_full_url(script):
    """Link convention (docs/m5-protocol.md): ?ws= is always a full ws:// URL."""
    text = PROTOCOL.read_text(encoding="utf-8")
    assert "화면 간 링크 규약" in text, "the link convention must be in the protocol doc"
    fn = re.search(r"function liveUrl\((.*?)\n\}", script, re.S).group(1)
    assert 'q.get("ws")' in fn
    assert "wss?:" in fn, "the value must be checked for a ws:// or wss:// prefix"
    assert "무시" in fn, "a value that is not a full URL is ignored, not patched up"


def test_hello_assets_is_used(script):
    hello = re.search(r"function onLiveHello\((.*?)\n\}", script, re.S).group(1)
    assert "msg.assets" in hello
    loader = re.search(r"async function loadFromServer\((.*?)\n\}", script, re.S).group(1)
    assert "assets.skeleton_sets" in loader, "the advertised bundle list must be used"
    assert "assets.neurons_3d" in loader


def test_live_reconnects_like_the_cockpit(script):
    assert "const RECONNECT_MS = 2000;" in script
    assert "setTimeout(connectLive, RECONNECT_MS)" in script


def test_live_and_file_playback_do_not_both_drive_activity(script):
    """Only one source of activity at a time, or the readouts fight each other."""
    upd = re.search(r"function updateReadouts\((.*?)\n\}", script, re.S).group(1)
    assert "if(!S.live && S.run)" in upd
    st = re.search(r"function setLiveStatus\((.*?)\n\}", script, re.S).group(1)
    assert 'el("playBtn").disabled = on' in st and "setPlaying(false)" in st


def test_file_playback_still_works(script, html):
    """Regression: the run.json path from M6c is untouched."""
    assert "function loadRun" in script and "function readRunFile" in script
    assert 'id="scrub"' in html and 'id="playBtn"' in html
    assert "function advancePlayback" in script


# ------------------------------------------------------------- picking
def test_picking_exists_and_reports_identity(script, html):
    assert "function pickNeuron" in script and "function selectNeuron" in script
    assert 'id="pickInfo"' in html
    sel = re.search(r"function selectNeuron\((.*?)\n\}", script, re.S).group(1)
    for field in ("bodyId", "인덱스", "집합", "좌표 출처", "복셀", "μm"):
        assert field in sel, f"the panel should show {field}"
    assert "SRC_NAMES" in sel, "coordinate source comes from the contract's src codes"


def test_drag_does_not_select(script):
    """Rotating the view must not count as a click."""
    assert "moved: false" in script
    assert "S.drag.moved = true" in script
    assert "!S.drag.moved" in script


def test_picking_skips_hidden_sets(script):
    fn = re.search(r"function pickNeuron\((.*?)\n\}", script, re.S).group(1)
    assert "S.setVisible[c.setIdx[d]]" in fn
    assert "drawToNeuron" in fn, "picking returns a graph neuron index, not a draw index"


# ------------------------------------------------- downstream / skeleton highlight
def test_hops_needs_a_server(script, html):
    assert 'id="hopsBtn"' in html and 'id="hopsClearBtn"' in html
    assert "서버가 필요하다" in script, "say why the button is off in file mode"
    st = re.search(r"function setLiveStatus\((.*?)\n\}", script, re.S).group(1)
    assert 'el("hopsBtn").disabled = !on' in st


def test_hop_highlight_has_its_own_attribute_and_clear(script):
    assert "layout(location=3) in float aHop;" in script, "highlight is a vertex attribute, not a rebuild"
    assert "function applyHops" in script and "function clearHops" in script
    assert "uHopDim" in script, "unhighlighted points must recede so the highlight reads"
    assert "function uploadHop" in script and "bufferSubData(GL.ARRAY_BUFFER, 0, S.hop)" in script


def test_skeleton_highlight_uses_contract_ranges(script):
    assert "S.skelIndex" in script
    add = re.search(r"function addSkel\((.*?)\n\}", script, re.S).group(1)
    assert "seg_offset" in add and "seg_count" in add
    render = re.search(r"function render\(ts\)\{(.*?)\n\}", script, re.S).group(1)
    assert "e.seg_offset * 2" in render and "e.seg_count * 2" in render, \
        "draw just that neuron's segment range"


# ------------------------------------------------------------- mock live server
def test_mock_live_server_speaks_the_same_protocol(script):
    assert "class MockLiveServer" in script
    assert "S.liveMock" in script
    mock = re.search(r"class MockLiveServer\{(.*?)\n\}\n", script, re.S).group(1)
    for t in ("hello", "frame", "ack", "hops", "snapshot_done"):
        assert f'"{t}"' in mock, f"the mock server must be able to send {t}"
    assert "assets:" in mock, "the mock advertises assets like the real server"


# ═══════════════════════════════════════════════════════════════════
#  M7e — 게임 재생 모드 (분할 화면): 왼쪽 게임, 가운데 3D 뇌, 아래 타임라인
# ═══════════════════════════════════════════════════════════════════
import zipfile

GAMES_BASE = ROOT / "flysim" / "apps" / "games" / "base.py"
GAMES_BRAIN = ROOT / "flysim" / "apps" / "games" / "brain.py"
GAMES_M7C = ROOT / "flysim" / "apps" / "games" / "m7c.py"
MOTOR = ROOT / "flysim" / "probe" / "motor.py"
SETS_PY = ROOT / "flysim" / "probe" / "sets.py"
BUNDLE_DIR = ROOT / "runs" / "m7c"
REMATCH_DIR = ROOT / "runs" / "m7c2"              # M7e2: rematch bundles (warm-up, CountWindowDecoder)
BUNDLES = sorted(BUNDLE_DIR.glob("*-brain-seed10")) if BUNDLE_DIR.exists() else []
REMATCH_BUNDLES = sorted(REMATCH_DIR.glob("*-brain-seed10")) if REMATCH_DIR.exists() else []
ALL_BUNDLES = BUNDLES + REMATCH_BUNDLES
needs_bundles = pytest.mark.skipif(not BUNDLES, reason="M7c replay bundles not present (runs/ is gitignored)")
needs_rematch = pytest.mark.skipif(not REMATCH_BUNDLES, reason="M7c2 replay bundles not present (runs/ is gitignored)")
_bundle_id = lambda p: f"{p.parent.name}/{p.name}"


def _js_list(script, name):
    return re.findall(r'"([^"]+)"', re.search(rf"const {name} = \[(.*?)\];", script, re.S).group(1))


# ---------------------------------------------------------------- 파일 형태
def test_replay_is_a_mode_of_the_3d_viewer_not_a_second_renderer(script, html):
    """Brief: one renderer (M6d). The game view lives inside flysim-3d.html."""
    assert not (ROOT / "flysim-game.html").exists(), "a second viewer file would duplicate the renderer"
    assert 'id="gamePane"' in html and 'id="tlPane"' in html and 'id="gameCanvas"' in html
    assert "body.game-mode" in html
    assert script.count('getContext("webgl2"') == 1, "still exactly one WebGL2 context"


def test_npz_is_read_in_the_browser(script):
    """Decision: frames.npz is unpacked client-side (no extra producer output)."""
    assert 'new DecompressionStream("deflate-raw")' in script
    for sig in ("0x06054b50", "0x02014b50", "0x04034b50"):
        assert sig in script, f"zip record signature {sig} not parsed"
    assert "zip64는 지원하지 않는다" in script, "zip64 must be rejected loudly, not misread"
    npy = re.search(r"function parseNpy\((.*?)\n\}", script, re.S).group(1)
    assert "'descr'" in npy and "'shape'" in npy and "fortran_order" in npy


def test_bundle_files_and_arrays(script):
    assert set(_js_list(script, "GAME_NPZ_ARRAYS")) == {"frames", "t_ms", "actions", "loom"}
    loader = re.search(r"async function loadBundleUrl\((.*?)\n\}", script, re.S).group(1)
    for f in ("frames.npz", "run.json", "game.json"):
        assert f in loader


@pytest.mark.skipif(not ALL_BUNDLES, reason="no replay bundles present (runs/ is gitignored)")
@pytest.mark.parametrize("bundle", ALL_BUNDLES, ids=_bundle_id)
def test_real_bundles_fit_the_browser_parser(script, bundle):
    """The JS parser handles stored/deflate zip entries without zip64 and the dtypes listed
    in NPY_DTYPES. If the producer's npz ever changes shape, this fails before the browser does."""
    known = set(re.findall(r'"([<|][a-z]\d)"', re.search(r"const NPY_DTYPES = \{(.*?)\};", script, re.S).group(1)))
    with zipfile.ZipFile(bundle / "frames.npz") as z:
        names = {i.filename for i in z.infolist()}
        assert {"frames.npy", "t_ms.npy", "actions.npy", "loom.npy"} <= names
        for info in z.infolist():
            assert info.compress_type in (0, 8), f"{info.filename}: zip method {info.compress_type}"
            assert info.file_size < 0xFFFFFFFF and info.compress_size < 0xFFFFFFFF, "zip64 entry"
            head = z.read(info.filename)[:256].decode("latin1")
            descr = re.search(r"'descr':\s*'([^']+)'", head).group(1)
            assert descr in known, f"{info.filename}: dtype {descr} not handled by parseNpy"
            assert "'fortran_order': False" in head
    for f in ("run.json", "game.json"):
        assert (bundle / f).exists()


# ---------------------------------------------------------------- 하네스와 같은 규칙
def test_actions_match_the_game_action_enum(script):
    src = GAMES_BASE.read_text(encoding="utf-8")
    body = re.search(r"class Action\(IntEnum\):(.*?)\n\n", src, re.S).group(1)
    enum = sorted(re.findall(r"(\w+) = (\d+)", body), key=lambda kv: int(kv[1]))
    assert _js_list(script, "GAME_ACTIONS") == [k for k, _ in enum]


def test_every_event_type_the_harness_emits_is_known(script):
    src = GAMES_M7C.read_text(encoding="utf-8")
    emitted = set(re.findall(r'"type":\s*"([a-z_]+)"', src)) | set(re.findall(r'"(crash|pass)" if', src)) \
        | set(re.findall(r'else "(crash|pass)"', src)) | set(re.findall(r'"(move_left|move_right)"', src))
    assert emitted, "could not read event types from the harness"
    known = set(_js_list(script, "GAME_EVENT_TYPES"))
    assert emitted <= known, f"replay does not know events {emitted - known}"
    assert "unknownEv" in script, "unknown event types are listed, not dropped silently"


def test_decoder_window_and_rule_match_the_harness(script):
    """M7e2: the decoder spec comes from the bundle (game.json agent.decoder), not a constant.
    The names the replay knows are the names motor.py's describe() writes, and the rule is the
    count rule of CountWindowDecoder (GiantFibreDecoder = the n = 1 case)."""
    assert "GAME_DECODER_MS" not in script, "no hard-coded decoder window any more"
    motor = MOTOR.read_text(encoding="utf-8")
    emitted = set(re.findall(r'"decoder": "(\w+)"', motor))
    known = set(_js_list(script, "GAME_DECODERS"))
    assert known == {"GiantFibreDecoder", "CountWindowDecoder"} and known <= emitted
    cw = re.search(r"class CountWindowDecoder:(.*?)\n\n\nclass |class CountWindowDecoder:(.*)", motor, re.S)
    cw = cw.group(1) or cw.group(2)
    assert '"window_ms": self.window_ms' in cw and '"min_spikes": self.min_spikes' in cw
    assert "c_L >= n only           -> RIGHT" in cw and "c_R >= n only           -> LEFT" in cw
    assert "lo <= t < now_ms" in cw, "window is [now - W, now)"
    gfd = re.search(r"class GiantFibreDecoder:(.*?)\n\n\nclass ", motor, re.S).group(1)
    assert "left side only          -> RIGHT" in gfd and "right side only         -> LEFT" in gfd
    # JUMP=1, LEFT=2, RIGHT=3: both -> JUMP, left only -> RIGHT, right only -> LEFT
    assert "const fl = wl.length >= spec.n, fr = wr.length >= spec.n;" in script
    assert "const a = fl && fr ? 1 : fl ? 3 : fr ? 2 : 0;" in script
    assert "const lo = tEnd[k] - win, hi = tEnd[k];" in script and "ms >= lo && ms < hi" in script


def _node_eval(script, fn_names, expr):
    """Run the named top-level functions of the page in node and return JSON of ``expr``."""
    import json as _json
    import shutil
    import subprocess
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed")
    src = []
    for name in fn_names:
        src.append(re.search(rf"\nfunction {name}\(.*?\n\}}", script, re.S).group(0))
    src.append(f"process.stdout.write(JSON.stringify({expr}));")
    out = subprocess.run([node, "-e", "\n".join(src)], capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    return _json.loads(out.stdout)


def test_both_decoder_specs_are_read_and_unknown_ones_switch_the_check_off(script):
    import json as _json
    cases = {
        "m7c": {"decoder": "GiantFibreDecoder", "window_ms": 50.0,
                "rule": "both GF -> JUMP; left GF only -> RIGHT; right GF only -> LEFT; none -> NOOP"},
        "m7c2": {"decoder": "CountWindowDecoder", "window_ms": 300.0, "min_spikes": 2},
        "motor": {"decoder": "MotorDecoder", "window_ms": 50.0, "min_rate_hz": 10.0},
        "no_n": {"decoder": "CountWindowDecoder", "window_ms": 300.0},
        "gf_n2": {"decoder": "GiantFibreDecoder", "window_ms": 50.0, "min_spikes": 2},
        "n_frac": {"decoder": "CountWindowDecoder", "window_ms": 300.0, "min_spikes": 1.5},
        "no_w": {"decoder": "CountWindowDecoder", "min_spikes": 2},
        "none": None,
    }
    got = _node_eval(script, ["decoderSpec", "decoderLine"],
                     "Object.fromEntries(Object.entries(" + _json.dumps(cases) + ").map(([k, d]) => "
                     "{ const s = decoderSpec(d); return [k, {...s, line: decoderLine(s, k.toUpperCase())}]; }))")
    assert got["m7c"] == {"known": True, "name": "GiantFibreDecoder", "W": 50, "n": 1, "line": "50 ms 안 한 발 (M7C)"}
    assert got["m7c2"] == {"known": True, "name": "CountWindowDecoder", "W": 300, "n": 2,
                           "line": "300 ms 안 같은 쪽 2발 (M7C2)"}
    for k in ("motor", "no_n", "gf_n2", "n_frac", "no_w", "none"):
        assert got[k]["known"] is False, k
        assert got[k]["line"].startswith("해독기 규격을 모름"), k
    assert got["motor"]["W"] == 50 and got["no_w"]["W"] is None and got["none"]["W"] is None
    # an unknown spec turns the recomputation off instead of guessing
    assert "const decoded = spec.known ? new Int8Array(T) : null;" in script
    stats = re.search(r"function fillGameStats\((.*?)\n\}", script, re.S).group(1)
    assert "해독기 규격을 모름 — 검사 끔" in stats


@pytest.mark.skipif(not ALL_BUNDLES, reason="no replay bundles present (runs/ is gitignored)")
def test_real_bundle_specs_tags_and_warmup(script):
    import json as _json
    rows = []
    for b in ALL_BUNDLES:
        game = _json.loads((b / "game.json").read_text())
        meta = _json.loads((b / "run.json").read_text())["meta"]
        meta = {k: meta.get(k) for k in ("run_id", "warmup_steps", "dt_ms", "engine_params")}
        rows.append({"dir": b.parent.name, "decoder": game["agent"]["decoder"], "meta": meta})
    got = _node_eval(script, ["decoderSpec", "decoderLine", "runTag", "warmupMs"],
                     _json.dumps(rows) + ".map(r => { const s = decoderSpec(r.decoder); "
                     "return {dir: r.dir, line: decoderLine(s, runTag(r.meta)), warm: warmupMs(r.meta)}; })")
    for r in got:
        if r["dir"] == "m7c":
            assert r["line"] == "50 ms 안 한 발 (M7c)" and r["warm"] == 0
        else:
            assert r["line"] == "300 ms 안 같은 쪽 2발 (M7c2)" and r["warm"] == 5000
    edge = _node_eval(script, ["warmupMs"], "[warmupMs({warmup_steps: 5000}), warmupMs({warmup_steps: 1500, dt_ms: 1}), "
                                            "warmupMs({warmup_steps: 0, dt_ms: 1}), warmupMs({})]")
    assert edge == [None, 1500, 0, 0], "steps without dt -> unknown length (band stays), never a guess"


def test_warmed_up_bundles_draw_no_transient_band(script, html):
    """M7e2: a bundle recorded after >= 2 s of warm-up has no post-reset transient on its clock."""
    assert "const RESET_TRANSIENT_MS = 2000;" in script
    assert "transient: !(warmMs >= RESET_TRANSIENT_MS)," in script
    tl = re.search(r"function renderTimelineStatic\((.*?)\n\}", script, re.S).group(1)
    i_skip, i_band = tl.index("if(!g.transient){"), tl.index("}else if(t0 < RESET_TRANSIENT_MS){")
    assert i_skip < i_band, "the band is drawn only in the transient branch"
    assert 'el("legendTransient").hidden = !S.game.transient;' in script
    assert 'id="legendTransient"' in html and 'id="legendWarmup"' in html
    note = re.search(r"function fillGameStats\((.*?)\n\}", script, re.S).group(1)
    assert "뒤부터 기록돼 리셋 직후 과도 구간(M7a2)의 발화는 들어 있지 않다" in note
    assert "묶음에 예열이 없어" in note, "the M7c wording stays for bundles without warm-up"
    info = re.search(r"function fillGameInfo\((.*?)\n\}", script, re.S).group(1)
    assert 'el("decoderLine")' in info and "M7c2 리포트 §4" in info


def test_classification_window_is_the_decoder_window(script, html):
    """Orange/violet uses the decoder's own window (W), so 50 ms is no longer written anywhere."""
    assert "const win = spec.W;" in script
    assert "50 ms" not in re.sub(r"<script.*?</script>", "", html, flags=re.S), "the static page still says 50 ms"
    stats = re.search(r"function fillGameStats\((.*?)\n\}", script, re.S).group(1)
    assert '"\\"입력 동반\\" = 스파이크 직전 " + g.win + " ms(해독기 창)' in stats
    assert '"램프는 해독기가 보는 창(직전 " + g.win + " ms)' in stats


@pytest.mark.skipif(not ALL_BUNDLES, reason="no replay bundles present (runs/ is gitignored)")
@pytest.mark.parametrize("bundle", ALL_BUNDLES, ids=_bundle_id)
def test_time_alignment_reproduces_the_recorded_actions(bundle):
    """The replay's alignment rule, re-implemented here: tick k is decided at the end of the tick
    from the giant-fibre spikes in [end - W, end), both sides >= n -> JUMP, one side -> away from
    it. W and n come from game.json agent.decoder (M7c: 50 ms, 1; M7c2: 300 ms, 2). The spikes come
    from the one-neuron escape regions of run.json. Every recorded action must come out the same
    (M7c 574/574 and 196/196, M7c2 196/196)."""
    import json as _json
    import numpy as np
    run = _json.loads((bundle / "run.json").read_text())
    game = _json.loads((bundle / "game.json").read_text())
    arr = np.load(bundle / "frames.npz")
    t, actions = arr["t_ms"], arr["actions"]
    dur = game.get("bundle_ms") or game["duration_ms"]
    t_end = np.append(t[1:], dur)
    meta = run["meta"]
    gf = {}
    for side in ("L", "R"):
        r = run["regions"].index(f"DNp01_{side}")
        rate = np.asarray(run["frames"]["rates"][r])
        count = np.rint(rate * meta["rate_norm_hz"] * meta["region_sizes"][f"DNp01_{side}"] * meta["bin_ms"] / 1000)
        gf[side] = np.repeat(np.arange(len(rate)) * meta["bin_ms"], count.astype(int))
    dec = game["agent"]["decoder"]
    assert dec["decoder"] in ("GiantFibreDecoder", "CountWindowDecoder"), dec
    win, n = dec["window_ms"], dec.get("min_spikes", 1)
    got = []
    for e in t_end:
        l = ((gf["L"] >= e - win) & (gf["L"] < e)).sum() >= n
        r = ((gf["R"] >= e - win) & (gf["R"] < e)).sum() >= n
        got.append(1 if l and r else 3 if l else 2 if r else 0)
    assert np.array_equal(np.asarray(got), actions), "replay timing does not reproduce the harness"


# ---------------------------------------------------------------- 정직성 표시
def test_shortcut_is_stated_on_screen(html):
    assert "감지기는 직접 짠 것이고, 감지기부터 거대 신경섬유까지만 배선도다" in html
    assert "직접 짠 것" in html  # timeline rows label the detector rows too


def test_spontaneous_firing_is_shown_not_hidden(script, html):
    assert "자발" in html, "spontaneous giant-fibre spikes must be labelled"
    assert "C_GF_SPONT" in script and "C_GF_INPUT" in script
    assert "gfKind" in script and "quietRate" in script
    assert "입력 있을 때" in script and "입력 없을 때" in script, \
        "rates with and without input side by side, so co-occurrence is not read as cause"
    assert "입력이 그 스파이크를 일으켰다는 뜻이 아니다" in html
    assert "1.9 Hz" in script, "the no-input control rate from M7c is cited next to the measured one"


def test_events_and_actions_are_drawn(script):
    rows = [r for r in re.findall(r'\{key: "(\w+)"', re.search(r"const TL_ROWS = \[(.*?)\];", script, re.S).group(1))]
    for key in ("loomL", "loomR", "gfL", "gfR", "ttL", "ttR", "act", "ev"):
        assert key in rows, f"timeline row {key} missing"
    assert "GAME_EVENT_KO" in re.search(r"function updateGame\((.*?)\n\}", script, re.S).group(1)


def test_ending_is_never_invented(script):
    """Until the producer fix, game.json had no crash event; the replay must not make one up."""
    fn = re.search(r"function endingText\((.*?)\n\}", script, re.S).group(1)
    assert 'e.type === "crash"' in fn and "g.doc.crashed" in fn and "loss_cause" in fn
    assert "종료 사유 없음" in fn


def test_run_wide_active_fraction_is_explained(script, html):
    assert 'id="runWide"' in html
    info = re.search(r"function fillGameInfo\((.*?)\n\}", script, re.S).group(1)
    assert "active_neurons" in info and "배경 잡음" in info and "WARN_RUNAWAY" in info


def test_header_carries_score_agent_seed_k_and_operating_point(script):
    info = re.search(r"function fillGameInfo\((.*?)\n\}", script, re.S).group(1)
    for label in ("점수", "에이전트", "시드", '"k"', "동작점"):
        assert label in info, f"game header lacks {label}"


# ---------------------------------------------------------------- 소속·시간·표시
def test_group_membership_comes_from_the_run(script):
    """M7e preparation decision: membership from run.json neurons.region/regions, the point
    cloud's set array only as the fallback."""
    fn = re.search(r"function applyRunGrouping\((.*?)\n\}", script, re.S).group(1)
    assert "regionOf.length === c.n" in fn and "c.baseSetIdx" in fn
    assert "applyRunGrouping(regions, raw.neurons && raw.neurons.region)" in script


def test_escape_profile_sets_have_colours(script):
    types = re.findall(r'"(\w+)"', re.search(r"ESCAPE_TYPES = \[(.*?)\]", SETS_PY.read_text(encoding="utf-8")).group(1))
    colours = re.search(r"const STAGE_COLOR = \{(.*?)\};", script, re.S).group(1)
    for t in types:
        assert re.search(rf"\b{t}:", colours), f"escape set {t} has no colour"


def test_game_frame_is_the_last_one_not_after_now(script):
    fn = re.search(r"function gameFrameAt\((.*?)\n\}", script, re.S).group(1)
    assert "t[mid] <= ms" in fn, "the shown game frame must be the last one at or before now"


def test_crash_on_the_final_tick_is_reachable(script):
    """The crash is stamped at duration_ms, one bin after the clock's last bin."""
    fn = re.search(r"function updateGame\((.*?)\n\}", script, re.S).group(1)
    assert "atEnd ? Math.max(ms, g.dur) : ms" in fn


def test_paused_frame_holds_and_glow_is_skeleton_based(script):
    assert "const decaying = S.live || S.playing || !S.run;" in script
    assert "function drawSkeletonGlow" in script and "progGlow" in script
    glow = re.search(r"function drawSkeletonGlow\((.*?)\n\}", script, re.S).group(1)
    assert "seg_offset * 2" in glow and "seg_count * 2" in glow


def test_hidden_attribute_wins_over_display_rules(html):
    """.conn{display:flex} used to override hidden, so '연결 안 됨' showed in file mode."""
    assert "[hidden]{display:none!important}" in html


def test_static_assets_path_is_relative_only(script):
    boot = re.search(r"function boot\((.*?)\nboot\(\);", script, re.S).group(1)
    assert 'q.get("assets")' in boot and 'q.get("bundle")' in boot
    assert boot.count('includes("..")') >= 2, "asset/bundle paths must not climb out of the served tree"
