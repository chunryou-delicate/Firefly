"""M5a: flysim-live entry point — websocket control/telemetry + static file server.

    python -m flysim.live.server [--host 0.0.0.0] [--ws-port 8765] [--http-port 8080]
                                 [--synapse conductance] [--g ...] [--dt 1.0] [--mode rate]

The session runs whether or not anybody is connected (silence by default). Every client sees
the same frames and any client may steer — there is no lock and no authentication, so do not
expose these ports to a public network (see docs/m5a-report.md).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import signal
import sys
import time
from typing import Any

import numpy as np
import websockets
from websockets.asyncio.server import serve

from ..engine import EngineParams
from ..engine.params import DEFAULT_G_CONDUCTANCE
from . import protocol as P
from .export_neurons import build_context, write_neurons
from .session import DEFAULT_A_IN, RUNS_LIVE, LiveSession, compute_hops

CLIENT_QUEUE_MAX = 256


class Client:
    """One websocket with its own bounded outbox, so a slow browser cannot stall the others."""

    def __init__(self, ws: Any):
        self.ws = ws
        self.q: asyncio.Queue = asyncio.Queue(maxsize=CLIENT_QUEUE_MAX)
        self.dropped = 0
        self.task: asyncio.Task | None = None

    def offer(self, data: str, droppable: bool) -> None:
        try:
            self.q.put_nowait(data)
        except asyncio.QueueFull:
            if droppable:
                self.dropped += 1
                return
            try:                       # an ack/warning outranks a queued frame
                self.q.get_nowait()
                self.q.put_nowait(data)
            except (asyncio.QueueEmpty, asyncio.QueueFull):
                self.dropped += 1

    async def writer(self) -> None:
        try:
            while True:
                data = await self.q.get()
                await self.ws.send(data)
        except (asyncio.CancelledError, websockets.exceptions.ConnectionClosed):
            pass


class LiveServer:
    def __init__(self, session: LiveSession, loop: asyncio.AbstractEventLoop):
        self.session = session
        self.loop = loop
        self.clients: dict[Any, Client] = {}
        self.outq: asyncio.Queue = asyncio.Queue()
        self.frames_sent = 0

    # called from the simulation thread
    def emit(self, msg: dict, to: Any = None) -> None:
        try:
            self.loop.call_soon_threadsafe(self.outq.put_nowait, (msg, to))
        except RuntimeError:               # loop already closed during shutdown
            pass

    async def sender(self) -> None:
        while True:
            msg, to = await self.outq.get()
            data = json.dumps(msg, separators=(",", ":"))
            droppable = msg.get("type") == "frame"
            if droppable:
                self.frames_sent += 1
            targets = [self.clients[to]] if (to is not None and to in self.clients) else list(self.clients.values())
            for c in targets:
                c.offer(data, droppable)

    async def handler(self, ws: Any) -> None:
        client = Client(ws)
        self.clients[ws] = client
        client.task = asyncio.create_task(client.writer())
        try:
            await ws.send(json.dumps(self.session.hello(), separators=(",", ":")))
            async for raw in ws:
                if isinstance(raw, (bytes, bytearray)):
                    self._on_pcm(ws, raw)
                    continue
                await self._on_text(ws, raw)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.pop(ws, None)
            if client.task is not None:
                client.task.cancel()

    def _on_pcm(self, ws: Any, raw: bytes) -> None:
        audio = self.session._audio
        if audio is None:
            self.emit(P.error_msg("binary frame received but no audio stream is open "
                                  "(send audio_start first)"), ws)
            return
        if len(raw) % 4:
            self.emit(P.error_msg(f"binary frame of {len(raw)} bytes is not a whole number of "
                                  "float32 samples"), ws)
            return
        audio.push(np.frombuffer(raw, dtype="<f4"))

    async def _on_text(self, ws: Any, raw: str) -> None:
        try:
            obj = json.loads(raw)
        except Exception as e:
            self.emit(P.error_msg(f"undecodable JSON: {e}"), ws)
            return
        try:
            kind, msg = P.validate_client_message(obj)
        except P.ProtocolError as e:
            self.emit(P.error_msg(e.msg, e.req_id), ws)
            return
        except P.Reject as e:
            req = obj.get("req_id") if isinstance(obj.get("req_id"), int) else None
            self.emit(P.ack_msg(req, False, e.msg, self.session.params_version), ws)
            return
        if kind in ("get_hops", "snapshot"):
            asyncio.create_task(self._offthread(ws, kind, msg))
            return
        if kind == "audio_start":
            self.session.prepare_audio(msg["sample_rate"])     # accept PCM from this moment on
        self.session.submit(msg, ws)

    async def _offthread(self, ws: Any, kind: str, msg: dict) -> None:
        """get_hops and snapshot are heavy and read-only-ish: run them off the sim thread so
        pacing is not disturbed."""
        sess = self.session
        try:
            if kind == "get_hops":
                layers, trunc = await self.loop.run_in_executor(
                    None, compute_hops, sess.ctx.graph, msg["src"], msg["k"], msg["min_contacts"])
                sess.log_external({"kind": "control", "msg": msg, "ok": True,
                                   "result": f"layers={[len(l) for l in layers]} truncated={trunc}"})
                self.emit(P.ack_msg(msg["req_id"], True,
                                    f"hops: {[len(l) for l in layers]}"
                                    + (" (some layers truncated to the cap)" if any(trunc) else ""),
                                    sess.params_version), ws)
                self.emit(P.hops_msg(msg["req_id"], msg["src"], msg["k"], layers, trunc), ws)
            else:
                res = await self.loop.run_in_executor(None, sess.snapshot, msg["run_id"], msg["seconds"])
                self.emit(P.ack_msg(msg["req_id"], True, f"snapshot written to {res['path']}",
                                    sess.params_version), ws)
                self.emit(P.snapshot_done_msg(msg["req_id"], msg["run_id"], res["path"], res["n_frames"]), ws)
        except P.Reject as e:
            sess.log_external({"kind": "control", "msg": msg, "ok": False, "result": e.msg})
            self.emit(P.ack_msg(msg["req_id"], False, e.msg, sess.params_version), ws)
        except Exception as e:                                   # pragma: no cover - defensive
            sess.log_external({"kind": "control", "msg": msg, "ok": False, "result": repr(e)})
            self.emit(P.ack_msg(msg["req_id"], False, f"{type(e).__name__}: {e}", sess.params_version), ws)


def build_params(args) -> EngineParams:
    g = args.g
    if g is None:
        g = DEFAULT_G_CONDUCTANCE if args.synapse == "conductance" else None
    return EngineParams(synapse=args.synapse, g=g, dt=args.dt, noise_sigma=args.noise_sigma)


async def main_async(args) -> int:
    t0 = time.time()
    print(f"[live] loading graph, probe sets and coordinates …", flush=True)
    ctx = build_context()
    write_neurons(ctx)
    print(f"[live] context ready in {time.time() - t0:.1f} s: {ctx.n:,} neurons, "
          f"{len(ctx.sets.names)} regions, JO {len(ctx.jo_left)} L / {len(ctx.jo_right)} R", flush=True)

    loop = asyncio.get_running_loop()
    server: LiveServer | None = None
    session = LiveSession(ctx, params=build_params(args), device=args.device, bin_ms=args.bin_ms,
                          use_cuda_graph=args.cuda_graph,
                          seed=args.seed, a_in=args.a_in, mode=args.mode, speed=args.speed,
                          emit=lambda msg, to=None: server.emit(msg, to) if server else None)
    session.frame_every = args.frame_every
    server = LiveServer(session, loop)
    session.start()

    from .static import serve_static
    httpd = None
    if args.http_port:
        httpd = serve_static(args.host, args.http_port, args.ws_port)
        print(f"[live] http://{args.host}:{args.http_port}/  (cockpit page, neurons-v1.json)", flush=True)

    stop = asyncio.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:                              # pragma: no cover - non-POSIX
            pass
    sender = asyncio.create_task(server.sender())
    async with serve(server.handler, args.host, args.ws_port, max_size=1 << 22):
        print(f"[live] ws://{args.host}:{args.ws_port}/ — engine {session.engine.backend} on "
              f"{session.device}, dt {session.dt_ms} ms, bin {session.bin_ms} ms, "
              f"g {session.engine.params.g:g}, a_in {session.a_in:g}, mode {session.stream.mode}",
              flush=True)
        print("[live] no authentication: keep these ports off public networks.", flush=True)
        await stop.wait()
    sender.cancel()
    session.stop()
    if httpd is not None:
        httpd.shutdown()
    print(f"[live] pacing: {json.dumps(session.pacing_metrics())}", flush=True)
    print(f"[live] control log: {session.log_path}", flush=True)
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="flysim-live server (M5a)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--ws-port", type=int, default=8765)
    ap.add_argument("--http-port", type=int, default=8080, help="0 disables the static server")
    ap.add_argument("--synapse", default="conductance", choices=("current", "conductance"))
    ap.add_argument("--g", type=float, default=None)
    ap.add_argument("--dt", type=float, default=1.0, choices=(1.0, 0.1))
    ap.add_argument("--mode", default="rate", choices=("rate", "phase-lock"))
    ap.add_argument("--a-in", dest="a_in", type=float, default=DEFAULT_A_IN)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--bin-ms", type=float, default=1.0)
    ap.add_argument("--noise-sigma", type=float, default=0.0)
    ap.add_argument("--frame-every", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-cuda-graph", dest="cuda_graph", action="store_false",
                    help="disable CUDA-graph capture of the engine step (slower; use if capture misbehaves)")
    ap.set_defaults(cuda_graph=None)
    args = ap.parse_args(argv)
    try:
        return asyncio.run(main_async(args))
    except KeyboardInterrupt:                                    # pragma: no cover
        return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
