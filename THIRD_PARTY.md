# Third-party data and licenses

## MaleCNS v1.0 connectome

- Source: https://male-cns.janelia.org/download/
- Files: `gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/` (public HTTPS mirror at `https://storage.googleapis.com/flyem-male-cns/...`)
- Paper: Cell, 2026-09-03, DOI 10.1016/j.cell.2026.08.015
- Producers: FlyEM (HHMI Janelia), University of Cambridge, MRC LMB, Google Research
- License: CC-BY 4.0 — https://creativecommons.org/licenses/by/4.0/

Raw data is not committed to this repository. `data/` is gitignored; run
`python -m flysim.data.download` to reproduce. Hashes of every downloaded
file are recorded in `data-provenance/hashes.json`.
