"""M7b: write game frames as images a person can look at — numpy + the standard library only.

No image library is installed (and none may be added), so PNG is written by hand with ``zlib``:

* ``write_png``    — one grayscale PNG
* ``write_apng``   — an animated PNG (plays in Firefox / Chrome; most other viewers show only
                     the first frame)
* ``contact_sheet``— a grid of frames in one PNG

``annotate`` adds a status strip **below** the game frame for humans: score digits, the two
looming bars (left / right) and the chosen action. The strip is drawn only into exported
images; agents never see it.
"""
from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np

_PNG_SIG = b"\x89PNG\r\n\x1a\n"


def _chunk(tag: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)


def _ihdr(h: int, w: int) -> bytes:
    return _chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))     # 8-bit grayscale


def _idat_payload(img: np.ndarray) -> bytes:
    img = np.ascontiguousarray(img, dtype=np.uint8)
    raw = b"".join(b"\x00" + img[r].tobytes() for r in range(img.shape[0]))
    return zlib.compress(raw, 9)


def png_bytes(img: np.ndarray) -> bytes:
    if img.ndim != 2:
        raise ValueError("grayscale only")
    return _PNG_SIG + _ihdr(*img.shape) + _chunk(b"IDAT", _idat_payload(img)) + _chunk(b"IEND", b"")


def write_png(path: Path, img: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png_bytes(img))
    return path


def write_apng(path: Path, frames: list[np.ndarray], fps: int = 30) -> Path:
    if not frames:
        raise ValueError("no frames")
    h, w = frames[0].shape
    out = [_PNG_SIG, _ihdr(h, w), _chunk(b"acTL", struct.pack(">II", len(frames), 0))]
    seq = 0
    for i, f in enumerate(frames):
        if f.shape != (h, w):
            raise ValueError("all frames must have the same size")
        out.append(_chunk(b"fcTL", struct.pack(">IIIIIHHBB", seq, w, h, 0, 0, 1, fps, 0, 0)))
        seq += 1
        data = _idat_payload(f)
        if i == 0:
            out.append(_chunk(b"IDAT", data))
        else:
            out.append(_chunk(b"fdAT", struct.pack(">I", seq) + data))
            seq += 1
    out.append(_chunk(b"IEND", b""))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(out))
    return path


def upscale(img: np.ndarray, k: int) -> np.ndarray:
    return np.repeat(np.repeat(img, k, axis=0), k, axis=1)


# ---- a 3x5 digit font, so the score can be drawn without a font library ------------------
_DIGITS = {
    "0": "111101101101111", "1": "010110010010111", "2": "111001111100111",
    "3": "111001111001111", "4": "101101111001001", "5": "111100111001111",
    "6": "111100111101111", "7": "111001001001001", "8": "111101111101111",
    "9": "111101111001111",
}


def draw_number(canvas: np.ndarray, value: int, x: int, y: int, scale: int = 3, ink: int = 255) -> None:
    for i, ch in enumerate(str(int(value))):
        bits = np.array([int(b) for b in _DIGITS[ch]], dtype=bool).reshape(5, 3)
        glyph = upscale(bits.astype(np.uint8), scale).astype(bool)
        x0 = x + i * 4 * scale
        region = canvas[y:y + glyph.shape[0], x0:x0 + glyph.shape[1]]
        region[glyph[:region.shape[0], :region.shape[1]]] = ink


def annotate(frame: np.ndarray, score: int, loom: tuple[float, float], action: int,
             scale: int = 3, loom_full_scale: float = 0.02) -> np.ndarray:
    """Upscaled frame with a status strip under it (humans only)."""
    big = upscale(frame, scale)
    w = big.shape[1]
    strip = np.full((10 * scale, w), 20, dtype=np.uint8)
    draw_number(strip, score, 2 * scale, 2 * scale, scale=scale)
    # two looming bars (left, right), height ~ signal / full scale
    bar_h = 8 * scale
    for k, v in enumerate(loom):
        frac = float(np.clip(v / loom_full_scale, 0.0, 1.0))
        x0 = w // 2 - 6 * scale + k * 7 * scale
        strip[scale:scale + bar_h, x0:x0 + 5 * scale] = 60
        filled = int(round(frac * bar_h))
        if filled:
            strip[scale + bar_h - filled:scale + bar_h, x0:x0 + 5 * scale] = 230
    # four action boxes NOOP / JUMP / LEFT / RIGHT, the chosen one lit
    for k in range(4):
        x0 = w - (4 - k) * 6 * scale
        strip[3 * scale:7 * scale, x0:x0 + 4 * scale] = 230 if k == action else 70
    return np.vstack([big, strip])


def contact_sheet(frames: list[np.ndarray], cols: int = 8, gap: int = 2) -> np.ndarray:
    if not frames:
        raise ValueError("no frames")
    h, w = frames[0].shape
    rows = (len(frames) + cols - 1) // cols
    sheet = np.full((rows * (h + gap) + gap, cols * (w + gap) + gap), 0, dtype=np.uint8)
    for i, f in enumerate(frames):
        r, c = divmod(i, cols)
        y, x = gap + r * (h + gap), gap + c * (w + gap)
        sheet[y:y + h, x:x + w] = f
    return sheet
