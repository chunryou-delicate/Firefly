"""M2: LIF engine parameters.

Every value here is a MODELING ASSUMPTION — the connectome carries no membrane
constants, synaptic gains or time constants (CLAUDE.md §3.3). The only free
parameter is ``g``; its default comes from the sweep in docs/m2-report.md.

Model (current-based LIF, exponential synapse, exact integration over one step;
docs/m2-brief.md "모델" and docs/m2-report.md):

    i_syn <- i_syn * a_s + g * sum_{pre spiked at t-1} w_signed        a_s = exp(-dt/tau_syn)
    v     <- v_rest + (v - v_rest) * a_m + (1 - a_m) * i_ext + c_s * i_syn   a_m = exp(-dt/tau_m)
                                   (refractory neurons are held at v_reset)
    spike <- v >= v_thresh          ; then v <- v_reset, refractory for t_ref

with c_s = tau_syn / (tau_m - tau_syn) * (a_m - a_s), the exact solution of
tau_m dv/dt = v_rest - v + i_ext + i_syn(t) for an exponentially decaying
i_syn over the step (i_ext constant over the step). ``R`` (membrane
resistance) is fixed at 1 and absorbed into ``g``. All dt-dependent
coefficients are functions of exp(-dt/tau) only, so dt=1.0 ms and dt=0.1 ms
describe the same dynamics up to the step-boundary discretisation of spike
detection and the one-step synaptic delay.

M2b (docs/m2b-brief.md): conductance-based synapses, selected with
``synapse="conductance"``. Per-neuron state v, g_e, g_i (dimensionless
conductances in units of 1/ms, leak = 1/tau_m):

    g_e  <- g_e * exp(-dt/tau_e) + g * n_exc_contacts_in     (excitatory pres that spiked)
    g_i  <- g_i * exp(-dt/tau_i) + g * n_inh_contacts_in     (|w| of inhibitory pres that spiked)
    ge_bar = g_e * avg_e,  gi_bar = g_i * avg_i              (step-average conductance,
                                   avg_x = tau_x*(1-exp(-dt/tau_x))/dt -> 1 as dt -> 0)
    G    = 1/tau_m + ge_bar + gi_bar
    v_inf = ((v_rest + i_ext)/tau_m + ge_bar*E_exc + gi_bar*E_inh) / G
    v    <- v_inf + (v - v_inf) * exp(-dt*G)                 (exponential Euler, per-neuron G)

The conductance entering G and v_inf is the exact average of the exponentially
decaying g_x over the step (so the integrated conductance per step is exact for
any dt); the remaining discretisation error is the one of a piecewise-constant
G within a step. With the start-of-step value instead, dt = 1 ms overdrives by
~10 % relative to dt = 0.1 ms (tau_e = 5 ms) and spike times differ by > 1 ms.

i_ext enters as (v_rest + i_ext)/tau_m, i.e. with g_e = g_i = 0 a constant i_ext
drives v towards v_rest + i_ext exactly as in the current model. One gain g for
both signs (no inhibitory scale factor — no extra free parameter). Reversal
potentials bound v to [E_inh, E_exc] when i_ext >= 0. Everything is an ASSUMPTION;
the change is a model-structure fix (unbounded hyperpolarisation, no low-rate
state in the current model), not a parameter tuned to a result.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import asdict, dataclass

# From the pre-defined g sweep (docs/m2-report.md, data-provenance/m2-sweep.json):
# the smallest primary-grid point that is VALID (normal range) for every tested
# stimulus seed (0, 1, 2). It lies in the ignited, self-sustained regime — see the
# report before relying on it for M3. Set once from the sweep; do not tune by hand.
DEFAULT_G = 0.886

# M2b conductance gain (1/ms per synaptic contact per spike). Rule fixed before the
# sweep (docs/m2b-brief.md): geometric mean of the RESPONSIVE g interval common to
# stimulus seeds 0, 1, 2 in data-provenance/m2b-sweep.json. Set once from the sweep.
DEFAULT_G_CONDUCTANCE = 3.162e-4   # sqrt(2.637e-4 * 3.793e-4), data-provenance/m2b-sweep.json (2026-09-14)

SYNAPSE_MODELS = ("current", "conductance")


@dataclass(frozen=True)
class EngineParams:
    dt: float = 1.0            # ms. 1.0 for rate mode, 0.1 for phase-lock mode (M3).
    tau_m: float = 20.0        # ms, membrane time constant           (ASSUMPTION)
    tau_syn: float = 5.0       # ms, synaptic current time constant   (ASSUMPTION)
    t_ref: float = 2.0         # ms, absolute refractory period       (ASSUMPTION)
    v_rest: float = -65.0      # mV                                   (ASSUMPTION)
    v_reset: float = -65.0     # mV                                   (ASSUMPTION)
    v_thresh: float = -50.0    # mV                                   (ASSUMPTION)
    g: float | None = None     # synaptic gain (ASSUMPTION; sweep). None -> DEFAULT_G / DEFAULT_G_CONDUCTANCE
                               #   current: mV per contact per spike; conductance: 1/ms per contact per spike
    noise_sigma: float = 0.0   # std of Gaussian current noise, referenced to dt = 1 ms; 0 = off
    v_floor: float | None = None  # optional hard lower clamp on v (mV); None = no clamp (default).
                                  # Unused (pointless) with conductance synapses: E_inh bounds v.
    # ---- M2b conductance synapses (all ASSUMPTIONS; ignored when synapse == "current") ----
    synapse: str = "current"   # "current" (M2 model, default) or "conductance" (M2b)
    E_exc: float = 0.0         # mV, excitatory reversal potential
    E_inh: float = -75.0       # mV, inhibitory reversal potential (GABA-A / GluCl)
    tau_e: float = 5.0         # ms, excitatory conductance decay
    tau_i: float = 10.0        # ms, inhibitory conductance decay (slower, GABA-A)

    def __post_init__(self) -> None:
        if self.synapse not in SYNAPSE_MODELS:
            raise ValueError(f"synapse must be one of {SYNAPSE_MODELS}, got {self.synapse!r}")
        if self.g is None:
            g = DEFAULT_G if self.synapse == "current" else DEFAULT_G_CONDUCTANCE
            if g is None:
                raise ValueError("no default g for the conductance model yet: pass g explicitly")
            object.__setattr__(self, "g", float(g))
        if self.g < 0:
            raise ValueError("g must be >= 0")
        if self.dt <= 0 or self.tau_m <= 0 or self.tau_syn <= 0 or self.tau_e <= 0 or self.tau_i <= 0:
            raise ValueError("dt, tau_m, tau_syn, tau_e, tau_i must be > 0")
        if not (self.E_inh <= self.v_reset and self.E_exc > self.v_thresh):
            raise ValueError("need E_inh <= v_reset and E_exc > v_thresh")
        if self.t_ref < 0 or self.noise_sigma < 0:
            raise ValueError("t_ref and noise_sigma must be >= 0")
        if not (self.v_reset < self.v_thresh):
            raise ValueError("need v_reset < v_thresh")
        if self.v_floor is not None and self.v_floor > self.v_reset:
            raise ValueError("v_floor must be <= v_reset")
        steps = self.t_ref / self.dt
        if abs(steps - round(steps)) > 1e-6:
            warnings.warn(f"t_ref/dt = {steps:.4f} is not an integer; refractory rounded to {round(steps)} steps")

    # ---- derived, dt-dependent coefficients (all exp(-dt/tau) form) -------
    @property
    def decay_m(self) -> float:
        return math.exp(-self.dt / self.tau_m)

    @property
    def decay_syn(self) -> float:
        return math.exp(-self.dt / self.tau_syn)

    @property
    def c_syn(self) -> float:
        """Coefficient of i_syn in the exact one-step membrane update (see module doc)."""
        if abs(self.tau_m - self.tau_syn) < 1e-9 * self.tau_m:
            # limit tau_syn -> tau_m: dt/tau_m * exp(-dt/tau_m)
            return self.dt / self.tau_m * self.decay_m
        return self.tau_syn / (self.tau_m - self.tau_syn) * (self.decay_m - self.decay_syn)

    @property
    def ref_steps(self) -> int:
        """Number of steps a neuron is held at v_reset after its spike step."""
        return int(round(self.t_ref / self.dt))

    @property
    def decay_e(self) -> float:
        return math.exp(-self.dt / self.tau_e)

    @property
    def decay_i(self) -> float:
        return math.exp(-self.dt / self.tau_i)

    @property
    def avg_e(self) -> float:
        """Step-average of exp(-t/tau_e) over one step (-> 1 as dt -> 0)."""
        return self.tau_e * (1.0 - self.decay_e) / self.dt

    @property
    def avg_i(self) -> float:
        return self.tau_i * (1.0 - self.decay_i) / self.dt

    @property
    def inv_tau_m(self) -> float:
        return 1.0 / self.tau_m

    @property
    def noise_scale(self) -> float:
        """Per-step current-noise std. ASSUMPTION: scaled by sqrt(1 ms / dt) so the
        stationary voltage variance driven by the noise is independent of dt."""
        return self.noise_sigma * math.sqrt(1.0 / self.dt)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(decay_m=self.decay_m, decay_syn=self.decay_syn, c_syn=self.c_syn, ref_steps=self.ref_steps,
                 noise_scale=self.noise_scale, decay_e=self.decay_e, decay_i=self.decay_i,
                 avg_e=self.avg_e, avg_i=self.avg_i)
        return d
