"""M0: download MaleCNS v1.0 flat-connectome files and verify integrity.

Files come from the public GCS bucket. The server returns MD5 in the
`x-goog-hash` header; we compute MD5 while streaming and refuse a file that
does not match. Every verified hash is recorded in data-provenance/hashes.json
so a re-run can prove the bytes are identical.

Usage: python -m flysim.data.download [--files name ...]
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

BASE_URL = (
    "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/"
)
ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROVENANCE = ROOT / "data-provenance" / "hashes.json"

# Core files for M0-M4. syn-partners (6.8 GB) and syn-points (12.7 GB) are
# deliberately excluded until a milestone actually needs them.
CORE_FILES = {
    "annotations": "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "neurotransmitters": "body-neurotransmitters-male-cns-v1.0.feather",
    "weights": "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
}
OPTIONAL_FILES = {
    "syn_partners": "syn-partners-male-cns-v1.0-minconf-0.5.feather",
    "body_stats": "body-stats-male-cns-v1.0-minconf-0.5.feather",
}
ALL_FILES = {**CORE_FILES, **OPTIONAL_FILES}


def _server_md5(headers) -> str | None:
    for v in headers.get_all("x-goog-hash") or []:
        if v.startswith("md5="):
            return base64.b64decode(v[4:]).hex()
    return None


def _load_provenance() -> dict:
    if PROVENANCE.exists():
        return json.loads(PROVENANCE.read_text())
    return {}


def _save_provenance(p: dict) -> None:
    PROVENANCE.parent.mkdir(parents=True, exist_ok=True)
    PROVENANCE.write_text(json.dumps(p, indent=2, sort_keys=True) + "\n")


def md5_of(path: Path, chunk=1 << 24) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def download(key: str, force: bool = False) -> Path:
    fname = ALL_FILES[key]
    dest = RAW_DIR / fname
    prov = _load_provenance()
    rec = prov.get(fname)

    if dest.exists() and rec and not force:
        local = md5_of(dest)
        if local == rec["md5"]:
            print(f"[skip] {fname}: exists, md5 matches provenance ({local})")
            return dest
        print(f"[warn] {fname}: local md5 {local} != recorded {rec['md5']}; re-downloading")

    url = BASE_URL + fname
    req = urllib.request.Request(url, headers={"User-Agent": "flysim/0.0.1"})
    tmp = dest.with_suffix(dest.suffix + ".part")
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    with urllib.request.urlopen(req) as resp:
        expected = _server_md5(resp.headers)
        total = int(resp.headers.get("Content-Length", 0))
        last_modified = resp.headers.get("Last-Modified")
        h = hashlib.md5()
        done = 0
        t0 = time.time()
        with tmp.open("wb") as f:
            while chunk := resp.read(1 << 22):
                f.write(chunk)
                h.update(chunk)
                done += len(chunk)
                if total and (done % (1 << 26) < (1 << 22) or done == total):
                    el = time.time() - t0
                    print(
                        f"\r[get ] {fname}: {done/1e6:8.1f}/{total/1e6:.1f} MB "
                        f"({100*done/total:5.1f}%) {done/1e6/max(el,1e-9):.1f} MB/s",
                        end="", file=sys.stderr, flush=True,
                    )
        print(file=sys.stderr)

    local = h.hexdigest()
    if expected is None:
        raise RuntimeError(f"{fname}: server sent no MD5 header; refusing to trust download")
    if local != expected:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"{fname}: MD5 mismatch local={local} server={expected}")
    if total and done != total:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"{fname}: size mismatch {done} != {total}")
    tmp.replace(dest)

    prov[fname] = {
        "url": url,
        "bytes": done,
        "md5": local,
        "server_last_modified": last_modified,
        "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    _save_provenance(prov)
    print(f"[ok  ] {fname}: {done} bytes md5={local}")
    return dest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--files", nargs="*", default=list(CORE_FILES), choices=list(ALL_FILES))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    for k in a.files:
        download(k, force=a.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
