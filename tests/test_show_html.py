"""M7e3 / M7e4: the show page ``flysim-show.html`` (docs/m7e3-brief.md section 4, docs/m7e4-brief.md section 3).

* Static checks always run: single file, no external scripts (one Google Fonts stylesheet), the
  honesty phrases, the contract fields it reads, the graph numbers against data-provenance.
* ``node`` checks the page's 84x84 eye drawing against ``base.background`` + ``fill_rect`` on
  many rectangles (skipped without node).
* Headless Chrome (DevTools protocol; skipped without google-chrome) opens the page twice: on the
  FAKE fixture from ``tests/show_fake.py`` and, when ``runs/show/`` exists, on the real files.
  Fonts are blocked there, so the page is also checked without them.
* M7e4 adds the arcade of eight classic games (``classic/<game>.json``, lazily loaded, and
  ``classic/index.json``) and the slow variants (``slow.json``): eye window vs ``check_frames`` and
  every recorded frame, lit buttons and HUD score vs the record on every tick, stamps and report
  card vs the contract's ``scores`` / ``verdict``, the honesty phrases, no original titles.
"""
from __future__ import annotations

import base64
import functools
import http.server
import json
import math
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
HTML = ROOT / "flysim-show.html"
REAL = ROOT / "runs" / "show"
sys.path.insert(0, str(Path(__file__).resolve().parent))


@pytest.fixture(scope="module")
def html() -> str:
    return HTML.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def script(html) -> str:
    blocks = re.findall(r"<script>(.*?)</script>", html, re.S)
    assert len(blocks) == 1, "one inline script"
    return blocks[0]


# ---------------------------------------------------------------- static
def test_single_file_no_external_scripts(html):
    assert not re.search(r"<script[^>]*\bsrc=", html), "no external scripts"
    links = re.findall(r"<link\b[^>]*>", html)
    sheets = [l for l in links if 'rel="stylesheet"' in l]
    assert len(sheets) == 1 and 'href="https://fonts.googleapis.com/css2?' in sheets[0]
    assert "family=Jua" in sheets[0] and "Noto+Sans+KR" in sheets[0]
    for l in links:
        if l not in sheets:
            assert 'href="data:' in l, f"only a data: icon besides the font sheet: {l}"
    assert "@import" not in html and not re.search(r"url\(\s*['\"]?https?:", html)
    assert not re.search(r"\b(fetch|import)\(\s*['\"]https?:", html)
    assert "<title>초파리 뇌 게임쇼</title>" in html


def test_fonts_fall_back_to_system_fonts(html):
    for fam in ('"Jua",', '"Noto Sans KR",'):
        assert fam in html
    assert "system-ui,sans-serif" in html.replace(" ", "")


def test_theme_tokens_for_light_and_dark(html):
    css = re.search(r"<style>(.*?)</style>", html, re.S).group(1)
    assert css.startswith("\n:root{")
    assert '@media (prefers-color-scheme: dark){\n  :root:not([data-theme="light"]){' in css
    assert ':root[data-theme="dark"]{' in css
    assert re.search(r"body\{[^}]*background:var\(--bg\)", css)
    # neuron colours shared with the M7e split screen: input cyan, input-accompanied orange, spontaneous purple
    for tok in ("--input:#3FBFCF", "--gfin:#E8A13A", "--spont:#B79BE0"):
        assert tok in css


def test_honesty_phrases(html):
    for phrase in ("초파리가 실제로 보는 화면", "가설 — 약하게만 지지됨(느린 판 실험)", "장식 그림 (기록 아님)", "착시 구간",
                   "꾸밈 요소", "커넥톰에 없는 것", "감지기", "가정",
                   # M7e4 (brief section 1 and 3.2)
                   "사후 점검", "사후 점검 — 판정 아님", "반칙", "정답을 아는 봇(반칙)", "총괄 읽기",
                   "어디까지 깰 수 있을까?", "화면을 보고 반응해서 이겼다고 말할 수 있는 게임은 아직 없음",
                   "같은 버튼 순서만으로도 점수가 나요. 상대를 보고 맞혔다고는 아직 말할 수 없어요.",
                   "시간만 늘려서는 못 이겼어요 · 원래 속도에서 신호만 세게 넣으면 이겼어요",
                   "통과의 절반쯤은 기둥이 나타나자마자 비킨 것", "(이 게임에선 효과 없음)"):
        assert phrase in html, phrase
    assert "가설 — 아직 확인 안 함" not in html, "the hypothesis tag was updated (brief 3.1.8)"


# the M7f brief's trademark list: the originals' full titles (tests/test_classic_games.py)
TRADEMARKS = ("pong", "breakout", "space invaders", "spaceinvaders", "invaders", "freeway", "kaboom",
              "demon attack", "demonattack", "skiing")


def test_no_original_titles_in_the_page(html):
    low = html.lower()
    for tm in TRADEMARKS:
        assert tm not in low, tm
    from test_classic_games import TRADEMARKS as theirs
    assert set(TRADEMARKS) == set(theirs), "keep the list the M7f tests use"


def test_reads_the_contract_fields(script):
    show = ["version", "meta", "fixture", "games", "title_ko", "rules", "eval_seeds", "scores", "fly_m7c",
            "fly_m7c2", "random", "one_way", "brainless", "one_way_dir", "brainless_theta", "ceiling", "verdict",
            "vs_random", "vs_one_way", "vs_brainless", "beats", "loses", "brainless_verdict",
            "tracks", "id", "game", "player", "match", "seed", "n_ticks", "score", "crashed", "truncated", "t_ms",
            "action", "loom", "state", "x", "obj_x", "dist", "height", "airborne", "speed", "eye_rect", "events",
            "tick", "type", "side", "obstacles", "spawn_tick", "end_tick", "outcome", "loom_last300_frac",
            "reach_t_ms", "theta", "brain", "decoder", "warmup_ms", "transient_ms", "gf", "ttmn", "set_rate_hz",
            "net_spikes", "check", "frames_bit_exact", "n_frames", "check_frames", "ticks", "raw_b64"]
    bmap = ["view", "screen_left", "box", "w", "h", "dots", "n", "xy", "sets", "n_neurons", "soma", "lines",
            "centroid", "generated"]
    for f in show + bmap:
        assert re.search(r"[.\"']" + re.escape(f) + r"\b", script), f"field {f!r} not read"
    for ev in ("spawn", "visible2px", "pass", "crash", "speedup", "score", "life_lost", "game_over"):
        assert f'"{ev}"' in script
    classic = ["genre_ko", "player_ko", "jump_effect", "left_effect", "right_effect", "kinds", "fixed_ref", "oracle",
               "yoked", "matrix_tape_x_game", "scene", "static", "shared", "shared_scene", "hud", "lives", "points",
               "check_frames", "k", "ceiling", "vs_fixed"]
    slow = ["rows", "family", "s", "lead_ms_median_top_k", "posthoc", "note", "cells", "cell", "sem",
            "reading_2x2", "reading_2x2_cells", "A_read_as_beat", "appearance", "within_300ms", "frac", "where",
            "hypothesis", "label", "reading", "speed_divisor", "cut_ms"]
    for f in classic + slow:
        assert re.search(r"[.\"']" + re.escape(f) + r"\b", script), f"field {f!r} not read"
    for name in ("classic/index.json", "slow.json", '"classic/" + g + ".json"'):
        assert name in script, name
    for name in ("LC4_L", "LC4_R", "LPLC2_L", "LPLC2_R", "DNp01_L", "DNp01_R", "TTMn_L", "TTMn_R"):
        assert f'"{name}"' in script


def test_eye_constants_are_base_py(script):
    from flysim.apps.games import base
    m = re.search(r"const EYE = \{W: (\d+), H: (\d+), HORIZON_ROW: ([\d.]+), SKY: ([\d.]+), GROUND: ([\d.]+), "
                  r"OBJECT: ([\d.]+)\};", script)
    assert m, "EYE constants"
    assert [float(x) for x in m.groups()] == [base.FRAME_W, base.FRAME_H, base.HORIZON_ROW, base.SKY,
                                              base.GROUND, base.OBJECT]
    assert "floor(k * 1000 / TICK_HZ)" in script and "const TICK_HZ = 60;" in script and base.TICK_HZ == 60


def test_graph_numbers_come_from_provenance(script):
    prov = json.loads((ROOT / "data-provenance" / "m1-build.json").read_text())
    m = re.search(r"const GRAPH = \{neurons: (\d+), edges: (\d+), source: \"([^\"]+)\"\};", script)
    assert m
    assert int(m.group(1)) == prov["neurons"]["retained"]
    assert int(m.group(2)) == prov["edges"]["retained"]
    assert m.group(3) == "data-provenance/m1-build.json"


def test_decoder_spec_is_the_split_screen_rule(script):
    other = (ROOT / "flysim-3d.html").read_text(encoding="utf-8")
    pick = lambda src: re.search(r"\nfunction decoderSpec\(d\)\{\n.*?\n\}\n", src, re.S).group(0)
    assert pick(script) == pick(other), "decoderSpec must stay the M7e2 rule"


def test_decorations_are_listed(script):
    decor = re.search(r"const DECOR = \[(.*?)\n\];", script, re.S).group(1)
    for word in ("날갯짓", "구름", "흔들림", "별", "속도선", "그림자", "기울기", "효과음", "장식 애니메이션",
                 # M7e4: what each kind is drawn as, the fly pilot, what goes outside a rectangle
                 "초파리 조종사", "장갑", "픽셀 그림", "도화선", "빛나는 공", "줄마다 다른 색", "색색의 차", "사각형 밖"):
        assert word in decor, word


def test_captions_say_signals_not_minds(script):
    bodies = [re.search(r"\nfunction " + fn + r"\(P(, add)?\)\{\n(.*?)\n\}\n", script, re.S).group(2)
              for fn in ("buildCaptions", "signalCaptions", "buildCaptionsClassic")]
    bodies.append(re.search(r"\nfunction effectKo\(G, act\)\{\n(.*?)\n\}\n", script, re.S).group(1))
    body = "\n".join(bodies)
    texts = " ".join(re.findall(r'"([^"]*)"', body))
    for bad in ("봤", "본다", "보았", "무서", "겁", "생각", "느끼", "느낀", "알아챘", "놀라", "원해", "싶어", "깨달",
                "노렸", "노려", "피하려", "맞히려", "잡으려", "알고"):
        assert bad not in texts, bad
    for good in ("감지 신호 ↑", "거대신경", "필요해요", "쾅!", "통과", "속도 UP!", "초파리 눈에 보이기 시작 (2픽셀)",
                 "점수 +", "목숨 −1", "게임 끝", "시간 끝", "발 → ", "(효과 없음)"):
        assert good in body, good


# ---------------------------------------------------------------- node: eye drawing vs base.py
def _node():
    node = shutil.which("node")
    if node is None:
        for cand in sorted(Path.home().glob(".nvm/versions/node/*/bin/node")):
            node = str(cand)
    return node


def _eye_js(script) -> str:
    pieces = [re.search(r"\nconst EYE = .*?;\n", script).group(0), re.search(r"\nconst clamp = .*?;\n", script).group(0),
              re.search(r"\nfunction roundHalfEven\(v\)\{.*?\}\n", script, re.S).group(0)]
    for fn in ("fillRectF32", "toFrame", "eyePixels", "classicPixels"):
        pieces.append(re.search(r"\nfunction " + fn + r"\(.*?\n\}\n", script, re.S).group(0))
    return "".join(pieces)


def test_eye_pixels_match_fill_rect(script, tmp_path):
    node = _node()
    if not node:
        pytest.skip("node not installed")
    from flysim.apps.games import base
    pieces = [_eye_js(script)]
    rng = np.random.default_rng(0)
    rects = [None, [42.0, 42.0, 30.0, 40.0], [-5.0, -1.0, 10.0, 20.0], [0.0, 84.0, 0.0, 84.0],
             [-30.0, 130.0, -20.0, 140.0], [41.5, 42.5, 35.5, 36.5], [10.25, 10.75, 50.125, 60.875]]
    for dist in (40.0, 30.1, 12.3, 3.7, 1.9, 1.6, 0.9):           # draw_block geometry, both games
        s = base.FOCAL_PX / dist
        for lat0, lat1, h in ((-0.8, 0.8, 0.6), (0.05, 1.05, 2.5), (-1.3, -0.3, 2.5)):
            rects.append([42 + s * lat0, 42 + s * lat1, 36 + s * (1 - h), 36 + s])
    for _ in range(400):
        x0, y0 = rng.uniform(-10, 90, 2)
        rects.append([float(x0), float(x0 + rng.uniform(0, 30)), float(y0), float(y0 + rng.uniform(0, 30))])
    for _ in range(100):                                           # half-integer edges: rint ties
        x0, y0 = rng.integers(-4, 88, 2) / 2
        rects.append([float(x0), float(x0 + rng.integers(1, 20) / 2), float(y0), float(y0 + rng.integers(1, 20) / 2)])
    src = tmp_path / "eye.js"
    src.write_text("".join(pieces) + "\nconst R = " + json.dumps(rects) + ";\n"
                   "process.stdout.write(JSON.stringify(R.map(r => Buffer.from(eyePixels(r)).toString('base64'))));\n")
    out = subprocess.run([node, str(src)], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    got = json.loads(out.stdout)
    bad = []
    for r, b64 in zip(rects, got):
        img = base.background()
        if r is not None:
            base.fill_rect(img, *r, base.OBJECT)
        if base64.b64decode(b64) != base.to_frame(img).tobytes():
            bad.append(r)
    assert not bad, f"{len(bad)}/{len(rects)} rectangles differ, first {bad[:3]}"


def test_classic_pixels_match_common_draw(script, tmp_path):
    """M7e4: the classic eye window = classic.common.draw (background 190, rectangles in order,
    values 110 / 25), on random scenes of up to 80 overlapping rectangles."""
    node = _node()
    if not node:
        pytest.skip("node not installed")
    from flysim.apps.games.classic.common import draw
    rng = np.random.default_rng(1)
    scenes = [[], [[0, 0.0, 84.0, 0.0, 84.0, 25.0]], [[0, 3.0, 81.0, 3.0, 4.0, 110.0], [1, 38.0, 46.0, 12.0, 20.0, 25.0]]]
    for _ in range(300):
        n = int(rng.integers(1, 80))
        sc = []
        for _ in range(n):
            x0, y0 = rng.uniform(-6, 88, 2)
            w, h = rng.uniform(0, 14, 2) if rng.random() < 0.8 else rng.integers(0, 12, 2) / 2
            sc.append([int(rng.integers(0, 4)), float(x0), float(x0 + w), float(y0), float(y0 + h),
                       float(110.0 if rng.random() < 0.4 else 25.0)])
        scenes.append(sc)
    src = tmp_path / "classic.js"
    src.write_text(_eye_js(script) + "\nconst R = " + json.dumps(scenes) + ";\n"
                   "process.stdout.write(JSON.stringify(R.map(r => Buffer.from(classicPixels([r])).toString('base64'))));\n")
    out = subprocess.run([node, str(src)], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    got = json.loads(out.stdout)
    bad = [i for i, (sc, b64) in enumerate(zip(scenes, got))
           if base64.b64decode(b64) != draw([{"rect": it[1:5], "value": it[5]} for it in sc]).tobytes()]
    assert not bad, f"{len(bad)}/{len(scenes)} scenes differ, first {bad[:3]}"


# ---------------------------------------------------------------- headless Chrome
def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass


class Browser:
    """Minimal DevTools-protocol client (websockets is a project dependency)."""

    def __init__(self, root: Path, query: str, width: int, height: int, mobile: bool, profile: Path):
        from websockets.sync.client import connect
        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", _free_port()),
                                                     functools.partial(_Quiet, directory=str(root)))
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        port = _free_port()
        self.proc = subprocess.Popen(
            [shutil.which("google-chrome"), "--headless", "--no-sandbox", "--hide-scrollbars", "--mute-audio",
             f"--remote-debugging-port={port}", f"--user-data-dir={profile}", f"--window-size={width},{height}",
             "about:blank"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(200):
            try:
                pages = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/json", timeout=1).read())
                break
            except Exception:
                time.sleep(0.1)
        else:
            raise RuntimeError("chrome did not start")
        self.ws = connect([p for p in pages if p["type"] == "page"][0]["webSocketDebuggerUrl"], max_size=None)
        self.n = 0
        self.exceptions: list[str] = []
        self.call("Runtime.enable")
        self.call("Network.enable")
        self.call("Network.setBlockedURLs", urls=["*fonts.googleapis.com*", "*fonts.gstatic.com*"])
        self.call("Emulation.setDeviceMetricsOverride", width=width, height=height, deviceScaleFactor=1,
                  mobile=mobile)
        self.call("Page.navigate", url=f"http://127.0.0.1:{self.httpd.server_address[1]}/flysim-show.html{query}")
        for _ in range(300):
            try:
                if self.js("window.SHOW && window.SHOW.loaded"):
                    break
            except AssertionError:
                pass
            time.sleep(0.1)
        else:
            raise RuntimeError("page never loaded")

    def call(self, method, **params):
        self.n += 1
        my = self.n
        self.ws.send(json.dumps({"id": my, "method": method, "params": params}))
        while True:
            m = json.loads(self.ws.recv(timeout=300))
            if m.get("method") == "Runtime.exceptionThrown":
                d = m["params"]["exceptionDetails"]
                self.exceptions.append(str((d.get("exception") or {}).get("description") or d.get("text")))
            if m.get("id") == my:
                if "error" in m:
                    raise RuntimeError(m["error"])
                return m.get("result", {})

    def js(self, expr):
        r = self.call("Runtime.evaluate", expression=expr, returnByValue=True, awaitPromise=True)
        if r.get("exceptionDetails"):
            raise AssertionError(json.dumps(r["exceptionDetails"])[:600])
        return r["result"].get("value")

    def close(self):
        try:
            self.ws.close()
        finally:
            self.proc.terminate()
            self.proc.wait(timeout=10)
            self.httpd.shutdown()


DATASETS = ["fake", "real"]


@pytest.fixture(scope="module", params=DATASETS)
def site(request, tmp_path_factory):
    if not shutil.which("google-chrome"):
        pytest.skip("google-chrome not installed")
    if request.param == "real":
        if not (REAL / "show-data.json").exists() or not (REAL / "brain-map.json").exists():
            pytest.skip("runs/show/ not built")
        root, query, data_dir = ROOT, "", REAL
    else:
        import show_fake
        root = tmp_path_factory.mktemp("showsite")
        shutil.copy(HTML, root / "flysim-show.html")
        data_dir = show_fake.write(root / "fake")
        query = "?data=fake/"
    b = Browser(root, query, 1440, 900, False, tmp_path_factory.mktemp("chrome"))
    data = json.loads((data_dir / "show-data.json").read_text())
    yield {"name": request.param, "b": b, "data": data, "map": json.loads((data_dir / "brain-map.json").read_text()),
           "dir": data_dir}
    b.close()


def test_loads_cleanly(site):
    b, data = site["b"], site["data"]
    assert b.js("SHOW.loaded") is True
    assert b.js("SHOW.S.problems") == []
    assert b.js("SHOW.S.tracks.map(P => P.id)") == [t["id"] for t in data["tracks"]]
    fake_banner = b.js("[...document.querySelectorAll('.banner.fake')].length")
    assert fake_banner == (1 if site["name"] == "fake" else 0)
    assert b.js("document.querySelectorAll('.banner.err').length") == 0
    assert b.js("document.getElementById('decorList').children.length") >= 7
    assert b.exceptions == []


def test_circuit_counts_come_from_the_map(site):
    sets = site["map"]["sets"]
    n = lambda k: sets[k]["n_neurons"]
    txt = site["b"].js("[...['cEye','cGf','cTt']].map(id => document.getElementById(id).textContent)")
    assert txt == [f"{n('LC4_L') + n('LPLC2_L')} (왼) · {n('LC4_R') + n('LPLC2_R')} (오)",
                   f"{n('DNp01_L')} (왼) · {n('DNp01_R')} (오)", f"{n('TTMn_L')} (왼) · {n('TTMn_R')} (오)"]


def test_eye_drawing_matches_check_frames(site):
    b = site["b"]
    res = b.js("""(() => SHOW.S.tracks.map(P => {
        const cf = P.tr.check_frames; let same = 0;
        cf.ticks.forEach((tk, i) => { const raw = atob(cf.raw_b64[i]), px = SHOW.eyePixels(P.rects[tk]);
          let ok = raw.length === px.length; for(let j = 0; ok && j < px.length; j++) if(raw.charCodeAt(j) !== px[j]) ok = false;
          if(ok) same++; });
        return [P.id, same, cf.ticks.length, P.eyeCheck.ok]; }))()""")
    assert res and all(s == n == e for _, s, n, e in res), res
    # the eye window on screen (canvas read-back), fly tracks, at their check ticks
    for t in site["data"]["tracks"]:
        if t["player"] != "fly":
            continue
        for tk, raw in zip(t["check_frames"]["ticks"], t["check_frames"]["raw_b64"]):
            got = b.js(f"SHOW.stageLoad({json.dumps(t['id'])}); SHOW.stageTick({tk}); SHOW.eyeCanvasBytes()")
            assert bytes(got) == base64.b64decode(raw), (t["id"], tk)


def _fnv(frame: np.ndarray) -> int:
    h = 2166136261
    for v in frame.tobytes():
        h = ((h ^ v) * 16777619) & 0xFFFFFFFF
    return h


def test_eye_drawing_matches_every_recorded_frame(site):
    """Real data only: every frame of the fly bundles (and the recorded brainless episodes)."""
    if site["name"] != "real":
        pytest.skip("real bundles only")
    b = site["b"]
    pairs = []
    for t in site["data"]["tracks"]:
        if t["player"] == "fly":
            pairs.append((t["id"], ROOT / "runs" / t["match"] / f"{t['game']}-brain-seed10" / "frames.npz"))
        elif t["player"] == "brainless":
            sub = "m7b-v2" if t["game"] == "dodge_v2" else "m7b"
            pairs.append((t["id"], ROOT / "runs" / sub / f"{t['game']}-threshold-seed10.npz"))
    checked = 0
    for tid, npz in pairs:
        if not npz.exists():
            continue
        frames = np.load(npz)["frames"]
        got = b.js(f"SHOW.eyeHashes({json.dumps(tid)})")
        assert len(got) == len(frames), tid
        want = [_fnv(f) for f in frames]
        bad = [k for k, (g, w) in enumerate(zip(got, want)) if g != w]
        assert not bad, f"{tid}: {len(bad)} frames differ, first {bad[:5]}"
        checked += len(frames)
    assert checked > 0
    print(f"eye drawing == recorded frames on {checked} frames")


def test_buttons_follow_the_recorded_actions(site):
    b, data = site["b"], site["data"]
    total = 0
    for t in data["tracks"]:
        if t["player"] != "fly":
            continue
        n = b.js(f"""(() => {{ SHOW.stageLoad({json.dumps(t['id'])}); const P = SHOW.S.byId.get({json.dumps(t['id'])}); let m = 0;
            for(let k = 0; k < P.T; k++){{ SHOW.stageTick(k); const s = SHOW.stagePad(), a = P.tr.action[k];
              if(s.tick === k && JSON.stringify(s.lit) === JSON.stringify(a ? [a] : [])) m++; }}
            return m; }})()""")
        assert n == t["n_ticks"], (t["id"], n)
        total += n
    # race panels: every track, every tick (DOM only, to keep it quick)
    games = sorted({t["game"] for t in data["tracks"]})
    race_total = 0
    for g in games:
        for m in ("m7c", "m7c2"):
            res = b.js(f"""(() => {{ SHOW.raceSet({json.dumps(g)}, {json.dumps(m)}); const ps = SHOW.S.race.panels.filter(p => p.P);
                const T = Math.max(...ps.map(p => p.P.T)); const ok = {{}}, bad = [];
                for(let k = 0; k < T; k++){{ SHOW.raceSeek(Math.floor(k * 1000 / 60), true);
                  SHOW.racePanels().forEach((r, i) => {{ const P = ps.find(p => p.P.id === r.track); if(!P || k >= P.P.T) return;
                    const a = P.P.tr.action[k], want = JSON.stringify(a ? [a] : []);
                    if(r.tick === k && JSON.stringify(r.lit) === want) ok[r.track] = (ok[r.track] || 0) + 1; else if(bad.length < 5) bad.push([r.track, k]); }}); }}
                return {{ok, bad}}; }})()""")
            assert res["bad"] == [], res["bad"]
            for tid, cnt in res["ok"].items():
                tr = next(t for t in data["tracks"] if t["id"] == tid)
                assert cnt == tr["n_ticks"], (tid, cnt)
                race_total += cnt
    print(f"buttons == recorded action: stage {total} ticks, race {race_total} panel-ticks ({site['name']})")


def test_race_drop_out_time_is_the_track_end(site):
    b, data = site["b"], site["data"]
    for g in sorted({t["game"] for t in data["tracks"]}):
        for m in ("m7c", "m7c2"):
            b.js(f"SHOW.raceSet({json.dumps(g)}, {json.dumps(m)})")
            for t in data["tracks"]:
                if t["game"] != g or (t["player"] == "fly" and t["match"] != m):
                    continue
                end = (t["n_ticks"] * 1000) // 60
                crash = [e["t_ms"] for e in t["events"] if e["type"] == "crash"]
                if t["crashed"]:
                    assert crash == [end], (t["id"], crash, end)
                before = b.js(f"SHOW.raceSeek({end - 1}, true); SHOW.racePanels()")
                after = b.js(f"SHOW.raceSeek({end}, true); SHOW.racePanels()")
                pb = next(p for p in before if p["track"] == t["id"])
                pa = next(p for p in after if p["track"] == t["id"])
                assert not pb["out"] and pb["outMs"] is None, t["id"]
                assert pa["out"] and pa["outMs"] == end, (t["id"], pa)
                label = "탈락" if t["crashed"] else "완주"
                assert pa["badge"] == f"{label} {end / 1000:.2f}초", pa["badge"]


def test_report_card_numbers_are_the_file_scores(site):
    b, data = site["b"], site["data"]
    got = b.js("""(() => { const out = {}; document.querySelectorAll('.rc-game').forEach(c => { const g = c.dataset.game; out[g] = {};
        c.querySelectorAll('.rc-row').forEach(r => { out[g][r.dataset.key] = {dots: [...r.querySelectorAll('.dot')].map(d => Number(d.dataset.score)),
          mean: r.querySelector('.mean') ? Number(r.querySelector('.mean').dataset.mean) : null}; }); }); return out; })()""")
    for g, G in data["games"].items():
        for key, sc in G["scores"].items():
            assert got[g][key]["dots"] == sc, (g, key)
            assert math.isclose(got[g][key]["mean"], sum(sc) / len(sc)), (g, key)
    if "dodge_v2" in data["games"] and data["games"]["dodge_v2"]["ceiling"] is not None:
        assert b.js("document.querySelector('.rc-game[data-game=dodge_v2] svg').textContent.includes('천장 "
                    + str(data["games"]["dodge_v2"]["ceiling"]) + "')")


def test_verdict_stamps_follow_the_file_flags(site):
    b, data = site["b"], site["data"]
    for g, G in data["games"].items():
        for m in ("m7c", "m7c2"):
            v = G["verdict"][m]
            txt = b.js(f"document.querySelector('.rc-game[data-game={g}] .stamp[data-match={m}]').textContent")
            assert ("뇌 없는 봇에 짐" in txt) == v["vs_brainless"]["loses"], (g, m, txt)
            assert ("한 방향 봇에도 짐" in txt) == v["vs_one_way"]["loses"], (g, m, txt)
            assert ("구별 안 됨" in txt) == (not v["vs_random"]["beats"] and not v["vs_random"]["loses"]), txt
        raw = b.js(f"document.querySelector('.rc-game[data-game={g}] .rc-raw').textContent")
        assert G["verdict"]["m7c"]["brainless_verdict"] in raw


def test_why_card_shows_the_file_fraction_and_the_hypothesis_tag(site):
    b, data = site["b"], site["data"]
    n = b.js("document.querySelectorAll('#whyChips button').length")
    obs = [o for t in data["tracks"] if t["player"] == "fly" and t["match"] == "m7c2"
           for o in t["obstacles"] if o["loom_last300_frac"] is not None]
    assert n == len(obs) > 0
    for i in range(n):
        frac = b.js(f"document.querySelectorAll('#whyChips button')[{i}].click(); Number(document.getElementById('whyNum').dataset.frac)")
        want = [o["loom_last300_frac"] for o in obs][i]
        assert frac == want
        assert b.js("document.getElementById('whyNum').textContent") == f"{round(want * 100)}%"
    assert b.js("document.querySelector('#whyCard .hyp').textContent") == "가설 — 약하게만 지지됨(느린 판 실험)"
    assert b.js("document.querySelector('#whyCard a.golink').getAttribute('href')") == "#slow"


def test_reset_transient_only_on_first_match(site):
    b, data = site["b"], site["data"]
    for t in data["tracks"]:
        if t["player"] != "fly":
            continue
        shown = b.js(f"SHOW.stageLoad({json.dumps(t['id'])}); SHOW.stageSeek(1000); !document.getElementById('transient').hidden")
        later = b.js("SHOW.stageSeek(2500); !document.getElementById('transient').hidden")
        assert shown == (t["brain"]["warmup_ms"] < 2000), t["id"]
        assert later is False


def test_crash_caption_and_decoder_replay(site):
    b, data = site["b"], site["data"]
    for t in data["tracks"]:
        if t["player"] != "fly":
            continue
        end = (t["n_ticks"] * 1000) // 60
        cap = b.js(f"SHOW.stageLoad({json.dumps(t['id'])}); SHOW.stageSeek({end}); document.getElementById('caption').textContent")
        if t["crashed"]:
            assert cap == f"쾅! {end / 1000:.2f}초 만에 충돌", cap
        assert b.js(f"SHOW.S.byId.get({json.dumps(t['id'])}).brain.match") == t["n_ticks"], t["id"]


def test_phone_width_has_no_horizontal_scroll(site, tmp_path_factory):
    if site["name"] != "real":
        pytest.skip("checked once, on the real files")
    m = Browser(ROOT, "", 360, 760, True, tmp_path_factory.mktemp("chrome-m"))
    try:
        assert m.js("document.documentElement.scrollWidth") <= 360
        if (REAL / "classic" / "boxing.json").exists():          # M7e4: classic stage, classic race, slow race
            m.js("SHOW.classicLoad('boxing').then(() => { SHOW.stageLoad('boxing/fly'); SHOW.raceSet('boxing', 'm7c2'); return 1; })")
            assert m.js("document.documentElement.scrollWidth") <= 360
            if (REAL / "slow.json").exists():
                m.js("SHOW.raceSet('dodge_v2_s4', 'm7c2'); 1")
                assert m.js("document.documentElement.scrollWidth") <= 360
        assert m.js("SHOW.S.problems") == []
        assert m.exceptions == []
    finally:
        m.close()



# ---------------------------------------------------------------- M7e4: arcade, slow variants
CLASSIC = ["paddle", "bricks", "formation", "crossing", "catch", "demons", "ski", "boxing"]


@pytest.fixture(scope="module")
def classic(site):
    """Every classic game file (the page loads them lazily; here all of them, through the page)."""
    d = site["dir"] / "classic"
    if not all((d / f"{g}.json").exists() for g in CLASSIC):
        pytest.skip("classic/<game>.json not built")
    files = {g: json.loads((d / f"{g}.json").read_text()) for g in CLASSIC}
    loaded = site["b"].js("SHOW.classicLoadAll()")
    assert sorted(loaded) == sorted(CLASSIC)
    return files


@pytest.fixture(scope="module")
def slow(site):
    f = site["dir"] / "slow.json"
    if not f.exists():
        pytest.skip("slow.json not built")
    return json.loads(f.read_text())


def _i10(seeds):
    return seeds.index(10)


def test_m7e4_files_load_cleanly(site, classic, slow):
    b = site["b"]
    assert b.js("SHOW.S.problems") == []
    assert b.js("SHOW.S.notes") == []
    assert b.exceptions == []
    want = [t["id"] for g in CLASSIC for t in classic[g]["tracks"]] + [t["id"] for t in slow["tracks"]]
    assert sorted(b.js("[...SHOW.S.byId.keys()].filter(id => SHOW.S.byId.get(id).kind === 'classic' || SHOW.S.slowTracks.includes(SHOW.S.byId.get(id)))")) == sorted(want)
    # contract addition 1: index.json's game blocks are the game files' blocks
    index = json.loads((site["dir"] / "classic" / "index.json").read_text())
    assert sorted(index["games"]) == sorted(CLASSIC)
    for g in CLASSIC:
        assert index["games"][g] == classic[g]["game"], g
        assert {t["player"] for t in classic[g]["tracks"]} == {"fly", "random", "fixed", "brainless"}
        i = _i10(classic[g]["game"]["eval_seeds"])
        for t in classic[g]["tracks"]:
            assert classic[g]["game"]["scores"][t["player"]][i] == t["score"], t["id"]


def test_classic_eye_window_matches_check_frames(site, classic, slow):
    b = site["b"]
    res = b.js("""(() => [...SHOW.S.byId.values()].filter(P => P.kind === 'classic').map(P => {
        const cf = P.tr.check_frames; let same = 0;
        cf.ticks.forEach((tk, i) => { const raw = atob(cf.raw_b64[i]), px = SHOW.classicPixels([P.scene.static, P.scene.at[tk]]);
          let ok = raw.length === px.length; for(let j = 0; ok && j < px.length; j++) if(raw.charCodeAt(j) !== px[j]) ok = false;
          if(ok) same++; });
        return [P.id, same, cf.ticks.length, P.eyeCheck.ok]; }))()""")
    assert len(res) == sum(len(classic[g]["tracks"]) for g in CLASSIC)
    assert all(s_ == n == e and n >= 3 for _, s_, n, e in res), res
    # the eye window on screen (canvas read-back), every classic track and the slow tracks, at their check ticks
    tracks = [t for g in CLASSIC for t in classic[g]["tracks"]] + slow["tracks"]
    for t in tracks:
        for tk, raw in zip(t["check_frames"]["ticks"], t["check_frames"]["raw_b64"]):
            got = b.js(f"SHOW.stageLoad({json.dumps(t['id'])}); SHOW.stageTick({tk}); SHOW.eyeCanvasBytes()")
            assert bytes(got) == base64.b64decode(raw), (t["id"], tk)
    print(f"classic eye window == check_frames on {sum(len(t['check_frames']['ticks']) for t in tracks)} frames ({site['name']})")


def test_classic_eye_window_matches_every_recorded_frame(site, classic, slow):
    """Real data only: every frame of the eight fly bundles (runs/m7f) and the slow fly run."""
    if site["name"] != "real":
        pytest.skip("real bundles only")
    b = site["b"]
    pairs = [(f"{g}/fly", ROOT / "runs" / "m7f" / f"{g}-brain-seed10" / "frames.npz") for g in CLASSIC]
    sfly = next(t for t in slow["tracks"] if t["player"] == "fly")
    for cand in (ROOT / "runs" / "m7e4" / "dodge_v2_s4-brain-seed10" / "frames.npz",
                 ROOT / "runs" / "m7g" / "dodge_v2_s4-brain-seed10" / "frames.npz"):
        if cand.exists():
            pairs.append((sfly["id"], cand))
            break
    checked = 0
    for tid, npz in pairs:
        assert npz.exists(), npz
        frames = np.load(npz)["frames"]
        got = b.js(f"SHOW.eyeHashes({json.dumps(tid)})")
        n = min(len(got), len(frames))
        assert n == len(got) or "m7g" in str(npz), (tid, len(got), len(frames))      # the M7g bundle holds the first 60 s
        bad = [k for k in range(n) if got[k] != _fnv(frames[k])]
        assert not bad, f"{tid}: {len(bad)} frames differ, first {bad[:5]}"
        checked += n
    print(f"classic + slow eye drawing == recorded frames on {checked} frames")


def test_classic_buttons_and_hud_follow_the_record(site, classic, slow):
    b = site["b"]
    stage = 0
    for g in CLASSIC:
        t = next(x for x in classic[g]["tracks"] if x["player"] == "fly")
        res = b.js(f"""(() => {{ SHOW.stageLoad({json.dumps(t['id'])}); const P = SHOW.S.byId.get({json.dumps(t['id'])}); let m = 0; const bad = [];
            for(let k = 0; k < P.T; k++){{ SHOW.stageTick(k); const s = SHOW.stagePad(), h = SHOW.stageHud(), a = P.tr.action[k];
              const ok = s.tick === k && JSON.stringify(s.lit) === JSON.stringify(a ? [a] : []) && h.shown && h.tick === k && h.hud === P.tr.hud.score[k]
                && h.score === P.tr.hud.score[k] && h.lives === (P.tr.hud.lives ? P.tr.hud.lives[k] : null);
              if(ok) m++; else if(bad.length < 5) bad.push(k); }}
            return {{m, bad}}; }})()""")
        assert res["bad"] == [] and res["m"] == t["n_ticks"], (t["id"], res)
        stage += res["m"]
    # race panels: every classic track and the slow tracks, every tick (DOM only): lit button and score = hud.score
    race = 0
    games = [(g, {x["id"]: x for x in classic[g]["tracks"]}, "hud") for g in CLASSIC]
    games.append(("dodge_v2_s4", {x["id"]: x for x in slow["tracks"]}, "state"))
    for g, byid, src in games:
        res = b.js(f"""(() => {{ SHOW.raceSet({json.dumps(g)}, 'm7c2'); const ps = SHOW.S.race.panels.filter(p => p.P);
            const T = Math.max(...ps.map(p => p.P.T)); const ok = {{}}, bad = [];
            for(let k = 0; k < T; k++){{ SHOW.raceSeek(Math.floor(k * 1000 / 60), true);
              SHOW.racePanels().forEach(r => {{ const p = ps.find(q => q.P.id === r.track); if(!p || k >= p.P.T) return;
                const a = p.P.tr.action[k], sc = {json.dumps(src)} === 'hud' ? p.P.tr.hud.score[k] : p.P.tr.state.score[k];
                if(r.tick === k && JSON.stringify(r.lit) === JSON.stringify(a ? [a] : []) && Number(r.shown) === sc) ok[r.track] = (ok[r.track] || 0) + 1;
                else if(bad.length < 5) bad.push([r.track, k]); }}); }}
            return {{ok, bad, keys: SHOW.racePanels().map(r => r.key)}}; }})()""")
        assert res["bad"] == [], (g, res["bad"])
        assert res["keys"] == ["fly", "random", "fixed", "brainless"], res["keys"]
        assert sorted(res["ok"]) == sorted(byid), (g, sorted(res["ok"]))
        for tid, cnt in res["ok"].items():
            assert cnt == byid[tid]["n_ticks"], (tid, cnt)
            race += cnt
    print(f"classic: buttons + hud == record on stage {stage} ticks (8 fly runs), race {race} panel-ticks "
          f"(32 classic + 4 slow runs) ({site['name']})")


def test_classic_and_slow_race_drop_out_at_the_track_end(site, classic, slow):
    b = site["b"]
    games = [(g, classic[g]["tracks"], True) for g in CLASSIC] + [("dodge_v2_s4", slow["tracks"], False)]
    for g, tracks, is_classic in games:
        b.js(f"SHOW.raceSet({json.dumps(g)}, 'm7c2')")
        for t in tracks:
            end = (t["n_ticks"] * 1000) // 60
            before = next(p for p in b.js(f"SHOW.raceSeek({end - 1}, true); SHOW.racePanels()") if p["track"] == t["id"])
            after = next(p for p in b.js(f"SHOW.raceSeek({end}, true); SHOW.racePanels()") if p["track"] == t["id"])
            assert not before["out"] and after["out"] and after["outMs"] == end, t["id"]
            if is_classic:
                label = "완주" if t["truncated"] else "게임 끝"
                assert int(after["shown"]) == t["score"]
            else:
                label = "탈락" if t["crashed"] else "완주"
            assert after["badge"] == f"{label} {end / 1000:.2f}초", after["badge"]


def test_cheating_bot_is_only_a_score_bar(site, classic):
    b = site["b"]
    for g in CLASSIC:
        G = classic[g]["game"]
        r = b.js(f"""(() => {{ SHOW.raceSet({json.dumps(g)}, 'm7c2'); const ob = document.getElementById('oracleBar');
            return {{keys: SHOW.racePanels().map(p => p.key), hidden: ob.hidden, text: ob.textContent,
                     score: Number(ob.querySelector('.obar[data-key=oracle]').dataset.score)}}; }})()""")
        assert r["keys"] == ["fly", "random", "fixed", "brainless"] and not r["hidden"]
        assert r["score"] == G["scores"]["oracle"][_i10(G["eval_seeds"])], g
        assert "정답을 아는 봇(반칙)" in r["text"]
    # first rung: no cheat bar
    b.js("SHOW.raceSet('dodge_v2', 'm7c2')")
    assert b.js("document.getElementById('oracleBar').hidden") is True


def _want(v, ceiling=False):
    return "beat" if v["beats"] else "lost" if v["loses"] else "ceil" if ceiling else "tie"


def _slow_ceiling(r):
    c = r["ceiling"]
    return bool(c["ceiling"]) if isinstance(c, dict) else c is not None


def test_ladder_and_arcade_stamps_follow_the_files(site, classic, slow):
    b = site["b"]
    b.js("buildLadder(); 1")
    cells = b.js("""[...document.querySelectorAll('#ladderMap .lcell')].map(c => ({zone: c.dataset.zone, game: c.dataset.game,
        v: Object.fromEntries([...c.querySelectorAll('.vw')].map(x => [x.dataset.vs, x.dataset.v]))}))""")
    want = []
    for g in ("jump", "dodge", "dodge_v2"):
        G = site["data"]["games"][g]
        v = G["verdict"]["m7c2"]
        want.append({"zone": "rung", "game": g, "v": {"random": _want(v["vs_random"]), "fixed": _want(v["vs_one_way"]),
                                                       "brainless": _want(v["vs_brainless"], G["ceiling"] is not None)}})
    rows = sorted(((n, r) for n, r in slow["rows"].items() if r["s"] > 1), key=lambda x: (x[1]["family"] != "dodge_v2", x[1]["s"]))
    for n, r in rows:
        v = r["verdict"]
        want.append({"zone": "slow", "game": n, "v": {"random": _want(v["vs_random"]), "fixed": _want(v["vs_fixed"]),
                                                       "brainless": _want(v["vs_brainless"], _slow_ceiling(r))}})
    for g in CLASSIC:
        G = classic[g]["game"]
        v = G["verdict"]
        want.append({"zone": "classic", "game": g, "v": {"random": _want(v["vs_random"]), "fixed": _want(v["vs_fixed"]),
                                                          "brainless": _want(v["vs_brainless"], G["ceiling"] is True)}})
    assert cells == want
    words = {"beat": "이김", "lost": "짐", "tie": "구별 안 됨", "ceil": "천장 — 판정 불가"}
    txt = b.js("[...document.querySelectorAll('#ladderMap .vw')].map(x => [x.dataset.v, x.textContent])")
    assert all(words[v] == t for v, t in txt), txt
    cabs = b.js("""[...document.querySelectorAll('#arcadeGrid .cab')].map(c => [c.dataset.game, c.querySelector('.marquee').textContent,
        c.querySelector('.genre').textContent, Object.fromEntries([...c.querySelectorAll('.vw')].map(x => [x.dataset.vs, x.dataset.v]))])""")
    assert [c[0] for c in cabs] == CLASSIC
    for g, title, genre, v in cabs:
        G = classic[g]["game"]
        assert title == G["title_ko"] and genre == G["genre_ko"]
        assert v == next(w["v"] for w in want if w["zone"] == "classic" and w["game"] == g)


def test_classic_report_card_numbers_are_the_file_scores(site, classic):
    b = site["b"]
    b.js("buildCardClassic(); 1")
    got = b.js("""(() => { const out = {}; document.querySelectorAll('#rcGridClassic .rc-game').forEach(c => { const g = c.dataset.game; out[g] = {rows: {}, st: {}};
        c.querySelectorAll('.rc-row').forEach(r => { out[g].rows[r.dataset.key] = {dots: [...r.querySelectorAll('.dot')].map(d => Number(d.dataset.score)),
          mean: r.querySelector('.mean') ? Number(r.querySelector('.mean').dataset.mean) : null}; });
        c.querySelectorAll('.stamp').forEach(s => { out[g].st[s.dataset.vs] = s.dataset.v; }); }); return out; })()""")
    for g in CLASSIC:
        G = classic[g]["game"]
        for key in ("fly", "random", "fixed", "brainless", "yoked"):
            sc = G["scores"][key]
            assert got[g]["rows"][key]["dots"] == pytest.approx(sc), (g, key)
            assert math.isclose(got[g]["rows"][key]["mean"], sum(sc) / len(sc)), (g, key)
        assert got[g]["rows"]["oracle"]["dots"] == []                                  # a bar, no dots
        assert math.isclose(got[g]["rows"]["oracle"]["mean"], sum(G["scores"]["oracle"]) / 10)
        v = G["verdict"]
        assert got[g]["st"] == {"random": _want(v["vs_random"]), "fixed": _want(v["vs_fixed"]),
                                "brainless": _want(v["vs_brainless"], G["ceiling"] is True), "yoked": _want(G["yoked"]["verdict"])}
    assert b.js("document.querySelector('#rcGridClassic .rc-game[data-game=boxing] .stamp[data-vs=yoked]').textContent.includes('사후 점검 — 판정 아님')")


def test_blind_card_is_the_yoked_matrix(site, classic):
    b = site["b"]
    b.js("buildBlind(); 1")
    G = classic["boxing"]["game"]
    M = G["yoked"]["matrix_tape_x_game"]
    cells = b.js("[...document.querySelectorAll('#blindMap .bm-cell')].map(c => [Number(c.dataset.r), Number(c.dataset.c), Number(c.dataset.v)])")
    assert cells == [[i, j, M[i][j]] for i in range(10) for j in range(10)]
    for i in range(10):
        assert M[i][i] == G["scores"]["fly"][i]                                         # the diagonal is the real game
    real, tape = b.js("[Number(document.getElementById('blindReal').dataset.mean), Number(document.getElementById('blindYoked').dataset.mean)]")
    assert math.isclose(real, sum(G["scores"]["fly"]) / 10) and math.isclose(tape, sum(G["scores"]["yoked"]) / 10)
    yv = G["yoked"]["verdict"]
    line = b.js("document.getElementById('blindLine').textContent")
    assert (line == "같은 버튼 순서만으로도 점수가 나요. 상대를 보고 맞혔다고는 아직 말할 수 없어요.") == (not yv["beats"] and not yv["loses"])
    rows = b.js("Object.fromEntries([...document.querySelectorAll('#blindTable tr[data-game]')].map(r => [r.dataset.game, r.querySelector('.vw').dataset.v]))")
    assert rows == {g: _want(classic[g]["game"]["yoked"]["verdict"]) for g in CLASSIC}
    assert "사후 점검 — 판정 아님" in b.js("document.getElementById('blindCard').textContent")


def test_slow_card_reads_slow_json(site, slow):
    b = site["b"]
    for fam in ("dodge_v2", "jump"):
        got = b.js(f"""(() => {{ SHOW.S.slowFam = {json.dumps(fam)}; buildSlow(); return [...document.querySelectorAll('#slowChart .sl-group')].map(g => ({{row: g.dataset.row,
            bars: Object.fromEntries([...g.querySelectorAll('.sl-bar')].map(r => [r.dataset.key, Number(r.dataset.mean)])),
            ceil: g.querySelector('.sl-ceil') ? Number(g.querySelector('.sl-ceil').dataset.ceiling) : null, v: g.querySelector('.sl-v').dataset.v}})); }})()""")
        rows = sorted(((n, r) for n, r in slow["rows"].items() if r["family"] == fam), key=lambda x: x[1]["s"])
        assert [x["row"] for x in got] == [n for n, _ in rows]
        for x, (n, r) in zip(got, rows):
            for key in ("fly", "random", "fixed"):
                assert math.isclose(x["bars"][key], sum(r["scores"][key]) / 10), (n, key)
            c = r["ceiling"]
            if isinstance(c, dict):
                want_c = (c["value"] if c["value"] is not None else sum(r["scores"]["oracle"]) / 10) if c["ceiling"] else None
            else:
                want_c = c
            assert (x["ceil"] is None) == (want_c is None) and (want_c is None or math.isclose(x["ceil"], want_c)), (n, x["ceil"], want_c)
            assert x["v"] == _want(r["verdict"]["vs_random"]), n
    ph = slow["posthoc"]
    assert ph["posthoc"] is True and ph["note"] == "사후 점검 — 판정 아님"
    dodge = [n for n, r in slow["rows"].items() if r["family"] == "dodge_v2" and r["s"] > 1]
    rc = ph.get("reading_2x2_cells") or {}
    claim1 = all(not rc[n]["A_read_as_beat"] for n in dodge) and all(rc[n]["B"] == "beat" for n in dodge)
    l1, c1 = b.js("[document.getElementById('slowLine1').textContent, document.getElementById('slowLine1').dataset.claim]")
    assert (c1 == "true") == claim1 and (l1 == "시간만 늘려서는 못 이겼어요 · 원래 속도에서 신호만 세게 넣으면 이겼어요") == claim1
    ap = [a["frac"] for a in ph["appearance"] if a["row"] in dodge and a["where"] == "selected_k_eval"]
    med = sorted(ap)[len(ap) // 2]
    l2 = b.js("document.getElementById('slowLine2').textContent")
    assert (l2 == "통과의 절반쯤은 기둥이 나타나자마자 비킨 것") == (0.35 <= med <= 0.65)
    assert "사후 점검 — 판정 아님" in b.js("document.getElementById('slowPost').textContent")
    assert b.js("document.querySelector('#slowHyp .hyp').textContent") == slow["hypothesis"]["label"] == "가설 — 약하게만 지지됨(느린 판 실험)"
    # the 4x slow run opens in the stage and the race
    fly = next(t for t in slow["tracks"] if t["player"] == "fly")
    assert b.js("document.getElementById('slowWatch').click(); SHOW.S.stage.P.id") == fly["id"]
    assert b.js("document.getElementById('slowRace').click(); SHOW.racePanels().map(p => p.track)") == \
        [next(t["id"] for t in slow["tracks"] if t["player"] == pl) for pl in ("fly", "random", "fixed", "brainless")]


def test_no_original_titles_on_the_rendered_page(site, classic, slow):
    b = site["b"]
    b.js("buildLadder(); buildCardClassic(); buildBlind(); buildSlow(); buildRaceGames(); 1")
    b.js("document.getElementById('honestBox').open = true")
    text = b.js("document.body.innerText").lower()
    for tm in TRADEMARKS:
        assert tm not in text, tm
    for g in CLASSIC:
        blob = json.dumps(classic[g]["game"], ensure_ascii=False).lower()
        for tm in TRADEMARKS:
            assert tm not in blob, (g, tm)


def test_classic_files_without_the_index_still_fill_the_page(site, tmp_path_factory):
    """Fake only: without classic/index.json the page reads the eight game files in turn."""
    if site["name"] != "fake":
        pytest.skip("fake fixture only")
    root = tmp_path_factory.mktemp("noindex")
    shutil.copy(HTML, root / "flysim-show.html")
    shutil.copytree(site["dir"], root / "fake")
    (root / "fake" / "classic" / "index.json").unlink()
    m = Browser(root, "?data=fake/", 1280, 900, False, tmp_path_factory.mktemp("chrome-ni"))
    try:
        assert any("index.json" in n for n in m.js("SHOW.S.notes"))
        for _ in range(300):
            if m.js("Object.keys(SHOW.S.classic.games).length") == 8:
                break
            time.sleep(0.1)
        time.sleep(0.5)
        assert m.js("SHOW.S.problems") == []
        v = m.js("[...document.querySelectorAll('#ladderMap .lcell[data-zone=classic] .vw')].map(x => x.dataset.v)")
        assert len(v) == 24 and "none" not in v
        assert m.exceptions == []
    finally:
        m.close()
