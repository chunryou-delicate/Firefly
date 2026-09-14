"""M4 tests: sweep design fixed, n_pulses rule, a_in equals M3 selection, tuning.json schema, HTML figure parses."""
import json
from html.parser import HTMLParser

import pytest

from flysim.apps import m3_click as M3
from flysim.apps import m4_ipi as M4


def test_sweep_design_is_fixed():
    assert M4.IPI_MS == [20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0]
    assert M4.G == 0.336 and M4.G == M3.G
    assert M4.A_IN == {"rate": 640.0, "phase-lock": 2560.0}
    assert M4.PRE_MS == 100.0 and M4.TRAIN_MS == 700.0 and M4.TOTAL_MS == 1000.0
    assert M4.STIM_WINDOW_MS == (100.0, 800.0) and M4.LATE_FROM_MS == 800.0
    assert M4.PULSE_KW == {"pulse_ms": 10.0, "carrier_hz": 200.0}
    assert M4.IGNITION_LATE_FRACTION == 0.01
    M4.check_a_in_matches_m3()          # raises if data-provenance/m3-results.json disagrees


@pytest.mark.parametrize("ipi,n", [(20, 35), (25, 28), (30, 23), (35, 20), (40, 17), (45, 15), (50, 14), (55, 12), (60, 11)])
def test_n_pulses_and_stimulus_length(ipi, n):
    assert M4.n_pulses_for(float(ipi)) == n
    st = M4.make_stimulus(float(ipi))
    assert st.duration_ms == 1000.0
    assert st.params["n_pulses"] == n and st.params["stim_onset_ms"] == 100.0
    assert st.params["stim_offset_ms"] <= 800.0
    env = st.envelope_1ms()
    assert env[:100].max() == 0 and env[800:].max() == 0 and env.max() == 1


def _tuning():
    if not M4.PROV_JSON.exists():
        pytest.skip("data-provenance/m4-tuning.json not generated yet")
    return json.loads(M4.PROV_JSON.read_text())


def test_tuning_json_schema():
    T = _tuning()
    for k in ("meta", "curves", "runs"):
        assert k in T
    m = T["meta"]
    for k in ("params", "adapter", "stimulus", "git_commit", "ipi_ms", "a_in", "g", "probe_sets"):
        assert k in m
    assert m["g"] == 0.336 and m["a_in"] == M4.A_IN and m["ipi_ms"] == M4.IPI_MS
    assert set(T["curves"]) == {"rate", "phase-lock"}
    for mode, sets in T["curves"].items():
        assert m["params"][mode]["g"] == 0.336 and m["params"][mode]["noise_sigma"] == 0
        assert m["adapter"][mode]["mode"] == mode and m["adapter"][mode]["a_in"] == M4.A_IN[mode]
        for base in M4.BASE_SETS:
            for side in ("L", "R"):
                c = sets[f"{base}_{side}"]
                assert c["ipi_ms"] == M4.IPI_MS
                for k in ("spikes_per_pulse", "rate_stim", "latency_ms"):
                    assert len(c[k]) == len(M4.IPI_MS)
                assert all(v >= 0 for v in c["spikes_per_pulse"])
    assert len(T["runs"]) == 18
    for r in T["runs"]:
        for k in ("mode", "ipi_ms", "run_id", "n_pulses", "n_spikes", "late_active_frac", "ignited", "viewer_failure", "per_set"):
            assert k in r
        assert r["n_pulses"] == M4.n_pulses_for(r["ipi_ms"])
        assert r["ignited"] is False


class _Svg(HTMLParser):
    def __init__(self):
        super().__init__()
        self.svgs = 0
        self.polylines = 0
        self.external = []
        self.tables = 0

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "svg" and a.get("class") == "panel":
            self.svgs += 1
        if tag == "polyline":
            self.polylines += 1
        if tag == "table":
            self.tables += 1
        for k in ("src", "href"):
            if k in a and a[k] and a[k].startswith(("http", "//")):
                self.external.append(a[k])
        if tag in ("script", "link", "img"):
            self.external.append(tag)


def test_tuning_html_parses_and_is_self_contained():
    if not M4.DOC_HTML.exists():
        pytest.skip("docs/m4-tuning.html not generated yet")
    p = _Svg()
    p.feed(M4.DOC_HTML.read_text())
    assert p.svgs == 2 * len(M4.BASE_SETS)          # one panel per set per mode
    assert p.polylines == 2 * p.svgs                 # L and R per panel
    assert p.tables >= 2 and p.external == []
    assert "35" in M4.DOC_HTML.read_text()


# ---- M4b: phase B (conductance model), docs/m4b-brief.md ---------------------------
def _tuning_b():
    if not M4.PROV_JSON_B.exists():
        pytest.skip("data-provenance/m4b-tuning.json not generated yet")
    return json.loads(M4.PROV_JSON_B.read_text())


def test_phase_b_parameters_fixed_and_schema():
    from flysim.engine.params import DEFAULT_G_CONDUCTANCE
    T = _tuning_b()
    m = T["meta"]
    assert m["synapse"] == "conductance" and m["g"] == DEFAULT_G_CONDUCTANCE == 3.162e-4
    assert m["ipi_ms"] == M4.IPI_MS and m["tag"] == "b"
    for mode in ("rate", "phase-lock"):
        p = m["params"][mode]
        assert p["synapse"] == "conductance" and p["g"] == 3.162e-4 and p["noise_sigma"] == 0 and p["v_floor"] is None
        assert (p["E_exc"], p["E_inh"], p["tau_e"], p["tau_i"]) == (0.0, -75.0, 5.0, 10.0)
        assert (p["tau_m"], p["t_ref"], p["v_rest"], p["v_reset"], p["v_thresh"]) == (20.0, 2.0, -65.0, -65.0, -50.0)
        assert m["adapter"][mode]["a_in"] == m["a_in"][mode] and mode in m["a_in_source"]
    # a_in traceable to the m3b re-selection (or the documented fallback)
    r = json.loads(M4.out_paths("b")["m3_results"].read_text())
    assert r["synapse"] == "conductance" and r["g"] == 3.162e-4
    for mode in ("rate", "phase-lock"):
        sel = r["modes"][mode]["selected_a_in"]
        if sel is None:
            assert m["a_in"][mode] == M4.A_IN_PHASE_A[mode] and "fallback" in m["a_in_source"][mode]
        else:
            assert m["a_in"][mode] == sel and "selected" in m["a_in_source"][mode]
        for s in r["modes"][mode]["sweep"]:
            assert not (s["verdict"] == "PASS" and s["ignited"])       # an ignited point is never a PASS
    assert len(T["runs"]) == 18
    for r_ in T["runs"]:
        assert r_["n_pulses"] == M4.n_pulses_for(r_["ipi_ms"]) and isinstance(r_["ignited"], bool)
    for mode in T["curves"]:
        for base in M4.BASE_SETS:
            for side in ("L", "R"):
                assert len(T["curves"][mode][f"{base}_{side}"]["spikes_per_pulse"]) == 9


def test_phase_b_html_parses():
    path = M4.out_paths("b")["doc_html"]
    if not path.exists():
        pytest.skip("docs/m4b-tuning.html not generated yet")
    p = _Svg()
    p.feed(path.read_text())
    assert p.svgs == 2 * len(M4.BASE_SETS) and p.polylines == 2 * p.svgs and p.external == []
    assert "phase B (conductance)" in path.read_text()
