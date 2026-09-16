"""M5a: the small static file server that ships the cockpit page and the neuron layout.

Only an explicit allow-list is served (no directory listing, no path traversal): the cockpit
page ``flysim-live.html`` (built by the M5b session — a placeholder is served until it
exists), the offline viewer, and ``neurons-v1.json``.
"""
from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from ..data.download import ROOT
from .export_neurons import NEURONS_JSON

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
    return {
        "/flysim-live.html": (ROOT / "flysim-live.html", "text/html; charset=utf-8"),
        "/flysim-viewer.html": (ROOT / "flysim-viewer.html", "text/html; charset=utf-8"),
        "/neurons-v1.json": (NEURONS_JSON, "application/json"),
    }


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
            if path not in routes:
                self._send(b"not found\n", "text/plain; charset=utf-8", 404)
                return
            target, ctype = routes[path]
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
