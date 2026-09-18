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
