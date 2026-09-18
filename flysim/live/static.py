"""M5a: the small static file server that ships the cockpit page and the neuron layout.

Only an explicit allow-list is served (no directory listing, no path traversal): the cockpit
page ``flysim-live.html`` (built by the M5b session — a placeholder is served until it
exists), the offline viewer, and ``neurons-v1.json``.

M6b adds the 3D assets of ``docs/m6-3d-contract.md``: the neuron point cloud
(``/neurons-3d.json`` + ``/neurons-3d.bin``) and the skeleton bundles
(``/skel/<name>.json`` + ``/skel/<name>.bin``, listed by ``/skel/index.json``). Bundle names
are the only dynamic part of the allow-list, so they are matched against a strict pattern and
the resolved file must still sit directly in ``data/cache`` — a path that escapes is a 404
like any other unknown path. Assets that have not been generated yet are 404, not an error.
"""
from __future__ import annotations

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from ..data.download import ROOT
from ..probe.export3d import CACHE_DIR, OUT_BIN, OUT_JSON, skeleton_bundles
from .export_neurons import NEURONS_JSON

JSON_CT = "application/json"
BIN_CT = "application/octet-stream"
SKEL_INDEX_PATH = "/skel/index.json"
# One path component, no separators, no leading dot: "pC1", "JO_AB_L", "WED-R" are fine.
SKEL_NAME_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,63}$")
SKEL_PATH_RE = re.compile(r"^/skel/(?P<name>[^/]+)\.(?P<ext>json|bin)$")

PLACEHOLDER = """<!doctype html><html lang="ko"><meta charset="utf-8">
<title>flysim-live</title>
<style>body{background:#10161D;color:#C9D3DD;font:15px/1.6 system-ui,sans-serif;margin:0;
padding:48px;max-width:46rem}code{color:#3FBFCF}a{color:#3FBFCF}</style>
<h1>flysim-live 서버가 돌고 있습니다</h1>
<p>조종석 페이지 <code>flysim-live.html</code>가 아직 저장소에 없습니다. 이 파일은 M5b 세션이 만듭니다.
저장소 루트에 파일이 생기면 이 주소를 새로고침하세요.</p>
<p>웹소켓 주소: <code id="ws"></code> · 뉴런 좌표: <a href="/neurons-v1.json">/neurons-v1.json</a>
· 결과 재생기: <a href="/flysim-viewer.html">/flysim-viewer.html</a></p>
<script>document.getElementById('ws').textContent =
  'ws://' + location.hostname + ':%WSPORT%/ws';</script>
</html>"""


def _routes() -> dict:
    """The fixed half of the allow-list: path -> (file, content type)."""
    return {
        "/flysim-live.html": (ROOT / "flysim-live.html", "text/html; charset=utf-8"),
        "/flysim-viewer.html": (ROOT / "flysim-viewer.html", "text/html; charset=utf-8"),
        "/flysim-3d.html": (ROOT / "flysim-3d.html", "text/html; charset=utf-8"),
        "/neurons-v1.json": (NEURONS_JSON, JSON_CT),
        "/neurons-3d.json": (OUT_JSON, JSON_CT),
        "/neurons-3d.bin": (OUT_BIN, BIN_CT),
    }


def resolve_skel(path: str, cache_dir: Path | None = None) -> tuple[Path, str] | None:
    """``/skel/<name>.json|bin`` -> the bundle file, or None if the path is not one.

    The name must be a single safe component and the resolved file must sit directly in
    ``cache_dir``; anything else returns None and the caller answers 404.
    """
    cache_dir = Path(CACHE_DIR if cache_dir is None else cache_dir)
    m = SKEL_PATH_RE.match(path)
    if not m or not SKEL_NAME_RE.match(m.group("name")):
        return None
    ext = m.group("ext")
    target = (cache_dir / f"skel-{m.group('name')}.{ext}").resolve()
    if target.parent != cache_dir.resolve():
        return None
    return target, (JSON_CT if ext == "json" else BIN_CT)


def make_handler(ws_port: int):
    routes = _routes()

    class Handler(BaseHTTPRequestHandler):
        server_version = "flysim-live"
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):     # quieter than the default stderr spam
            pass

        def _send(self, body: bytes, ctype: str, code: int = 200) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def do_HEAD(self):                      # noqa: N802
            self.do_GET()

        def do_GET(self):                       # noqa: N802
            path = self.path.split("?", 1)[0]
            if path == "/":
                path = "/flysim-live.html"
            if path == SKEL_INDEX_PATH:         # a listing, not a file: always 200
                body = json.dumps(skeleton_bundles(CACHE_DIR), separators=(",", ":")).encode()
                self._send(body, JSON_CT)
                return
            if path in routes:
                target, ctype = routes[path]
            else:
                skel = resolve_skel(path, CACHE_DIR)
                if skel is None:
                    self._send(b"not found\n", "text/plain; charset=utf-8", 404)
                    return
                target, ctype = skel
            if not Path(target).exists():
                if path == "/flysim-live.html":
                    self._send(PLACEHOLDER.replace("%WSPORT%", str(ws_port)).encode(),
                               "text/html; charset=utf-8")
                else:
                    self._send(f"{path} has not been generated yet\n".encode(),
                               "text/plain; charset=utf-8", 404)
                return
            self._send(Path(target).read_bytes(), ctype)

    return Handler


def serve_static(host: str, port: int, ws_port: int) -> ThreadingHTTPServer:
    """Start the static server on its own daemon thread and return it."""
    httpd = ThreadingHTTPServer((host, port), make_handler(ws_port))
    threading.Thread(target=httpd.serve_forever, name="flysim-live-http", daemon=True).start()
    return httpd
