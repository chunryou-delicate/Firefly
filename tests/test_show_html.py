"""M7e3: the show page ``flysim-show.html`` (docs/m7e3-brief.md section 4).

* Static checks always run: single file, no external scripts (one Google Fonts stylesheet), the
  honesty phrases, the contract fields it reads, the graph numbers against data-provenance.
* ``node`` checks the page's 84x84 eye drawing against ``base.background`` + ``fill_rect`` on
  many rectangles (skipped without node).
* Headless Chrome (DevTools protocol; skipped without google-chrome) opens the page twice: on the
  FAKE fixture from ``tests/show_fake.py`` and, when ``runs/show/`` exists, on the real files.
  Fonts are blocked there, so the page is also checked without them.
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
    for phrase in ("초파리가 실제로 보는 화면", "가설 — 아직 확인 안 함", "장식 그림 (기록 아님)", "착시 구간",
                   "꾸밈 요소", "커넥톰에 없는 것", "감지기", "가정"):
        assert phrase in html, phrase


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
    for ev in ("spawn", "visible2px", "pass", "crash", "speedup"):
        assert f'"{ev}"' in script
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
    for word in ("날갯짓", "구름", "흔들림", "별", "속도선", "그림자", "기울기", "효과음", "장식 애니메이션"):
        assert word in decor, word


def test_captions_say_signals_not_minds(script):
    body = re.search(r"\nfunction buildCaptions\(P\)\{\n(.*?)\n\}\n", script, re.S).group(1)
    texts = " ".join(re.findall(r'"([^"]*)"', body))
    for bad in ("봤", "본다", "보았", "무서", "겁", "생각", "느끼", "느낀", "알아챘", "놀라", "원해", "싶어", "깨달"):
        assert bad not in texts, bad
    for good in ("감지 신호 ↑", "거대신경", "필요해요", "쾅!", "통과", "속도 UP!", "초파리 눈에 보이기 시작 (2픽셀)"):
        assert good in body, good


# ---------------------------------------------------------------- node: eye drawing vs base.py
def _node():
    node = shutil.which("node")
    if node is None:
        for cand in sorted(Path.home().glob(".nvm/versions/node/*/bin/node")):
            node = str(cand)
    return node


def test_eye_pixels_match_fill_rect(script, tmp_path):
    node = _node()
    if not node:
        pytest.skip("node not installed")
    from flysim.apps.games import base
    pieces = [re.search(r"\nconst EYE = .*?;\n", script).group(0), re.search(r"\nconst clamp = .*?;\n", script).group(0)]
    for fn in ("roundHalfEven", "eyePixels"):
        pieces.append(re.search(r"\nfunction " + fn + r"\(.*?\n\}\n" if fn == "eyePixels"
                                else r"\nfunction " + fn + r"\(v\)\{.*?\}\n", script, re.S).group(0))
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
    yield {"name": request.param, "b": b, "data": data, "map": json.loads((data_dir / "brain-map.json").read_text())}
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
    assert b.js("document.querySelector('#whyCard .hyp').textContent") == "가설 — 아직 확인 안 함"


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
        assert m.js("SHOW.S.problems") == []
        assert m.exceptions == []
    finally:
        m.close()
