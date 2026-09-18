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

M2c (docs/m2c-brief.md): spike-frequency adaptation. One extra per-neuron state w
(an outward current, in the same units and the same place as i_ext), shared by both
synapse models:

    w <- w * exp(-dt/tau_w)                   every step, before the membrane update
    (the external-current slot becomes i_ext - w)
    w <- w + adapt_b                          for every neuron that spiked this step

so a neuron that fires accumulates its own hyperpolarising drive, which decays with
tau_w. tau_w = 100 ms is FIXED (ASSUMPTION; the usual range is 50-300 ms) and
adapt_b comes from the pre-registered sweep in docs/m2c-report.md. adapt_b = 0
(the default) disables the whole path, and the engine then runs the pre-M2c kernel
branch byte for byte (regression: data-provenance/m2-regression-spikes.npz).
Rationale, recorded before the sweep: both models are bistable (silent or ignited
at 150-245 Hz) with no low-rate state, because nothing makes a neuron's own firing
oppose itself. This adds that mechanism; it is not a parameter tuned to a result.

M2d (docs/m2d-brief.md): the same mechanism as a *conductance* instead of a current.
Per-neuron state g_a (1/ms, the same unit as the synaptic conductances):

    g_a  <- g_a * exp(-dt/tau_a)              every step, before the membrane update
    ga_bar = g_a * avg_a                      step-average, as for g_e / g_i
    G    = 1/tau_m + ge_bar + gi_bar + ga_bar
    v_inf = ((v_rest + i_ext)/tau_m + ge_bar*E_exc + gi_bar*E_inh + ga_bar*E_adapt) / G
    g_a  <- g_a + adapt_g_b                   for every neuron that spiked this step

This is the real mechanism (a potassium conductance), and unlike the M2c current it
is bounded by construction: it can only pull v towards E_adapt, never past it. M2c's
adaptation current drove v to -834 mV because a current has no reversal potential —
the same defect M2b fixed for the synapses. This is a model-structure fix of that
same kind, not a parameter tuned to a result.

Only defined for synapse == "conductance" (a current model has no reversal
potentials, so the concept is meaningless there) and mutually exclusive with the
M2c current-based adapt_b, which is kept at 0 for history.

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

# M2c adaptation increment per spike (same units as i_ext, i.e. mV of steady-state
# drive at R = 1). Rule fixed before the sweep (docs/m2c-brief.md): the smallest b
# on the grid at which the noise sweep reaches the pre-registered target state.
# Set once from data-provenance/m2c-noise-sweep.json; do not tune by hand.
#
# NOTE: this is NOT the dataclass default. ``EngineParams.adapt_b`` defaults to 0.0
# (adaptation off = the pre-M2c engine, bit for bit) exactly as the brief requires,
# so M3/M4/live call sites are unaffected until someone opts in by passing
# ``adapt_b=DEFAULT_ADAPT_B``. Whether to adopt it is a planning decision: at this b
# the adaptation current drives v to about -830 mV and the target state has 25-100 %
# of neurons active. See docs/m2c-report.md §4.3 and §5 before using it.
DEFAULT_ADAPT_B = 56.21452268581154   # data-provenance/m2c-noise-sweep.json (2026-09-16)

# M2d conductance-based adaptation increment per spike (1/ms, the synaptic conductance
# unit). Rule fixed before the sweep (docs/m2d-brief.md): the smallest b_g on the grid
# at which the pre-registered usable state exists. None = the sweep found none; see
# docs/m2d-report.md. Like DEFAULT_ADAPT_B this is NOT the dataclass default -
# EngineParams.adapt_g_b defaults to 0.0 so existing call sites are unaffected.
# M2e (docs/m2e-report.md) re-ran the same rule on a grid opened in both directions and
# resolved M2d's grid-edge answer: the minimum is now an INTERIOR point, 0.6579.
#
# DO NOT ADOPT THIS AS AN OPERATING POINT. The pre-registered robustness check failed:
# re-running the selected cell under noise seeds 3/4/5 gives active fractions 30.069 %,
# 29.954 % and 30.101 % against the <= 30 % condition, so 2 of 3 seeds fall outside. The
# cell clears the condition by only 0.263 points while the seed-to-seed spread is ~0.35,
# i.e. it sits inside the noise of the criterion. This is a consequence of the rule
# itself: "smallest b_g that passes" selects the least-robust passing point by
# construction, and headroom grows monotonically with b_g (0.26 pts here, 4.6 pts at
# b_g = 8.0). Changing the rule is a planning decision, not this window's.
DEFAULT_ADAPT_G_B = 0.6579332246575678         # data-provenance/m2e-noise-sweep.json (2026-09-19)
DEFAULT_G_WITH_ADAPT = 0.000536539159055945    # the g that goes with it (same rule)
DEFAULT_ADAPT_G_B_ROBUST = False               # data-provenance/m2e-robustness.json

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
    # ---- M2c spike-frequency adaptation (ASSUMPTIONS; both synapse models) ----
    adapt_b: float = 0.0       # increment of w per spike; 0 = off (pre-M2c path, bit-identical).
                               #   The sweep-selected value is DEFAULT_ADAPT_B — opt in explicitly.
    adapt_tau_w: float = 100.0         # ms, adaptation decay. FIXED by the brief; do not tune.
    # ---- M2d conductance-based adaptation (ASSUMPTIONS; conductance synapses only) ----
    adapt_g_b: float = 0.0     # increment of g_a per spike (1/ms); 0 = off (bit-identical).
                               #   The sweep-selected value is DEFAULT_ADAPT_G_B - opt in explicitly.
    adapt_tau_a: float = 100.0 # ms, adaptation conductance decay. FIXED by the brief (= tau_w, so
                               #   M2c and M2d are comparable); do not tune.
    E_adapt: float = -75.0     # mV, adaptation reversal potential (ASSUMPTION: = E_inh. The K+
                               #   reversal is usually -80..-90, but the brief keeps the constant
                               #   count down; exposed as a parameter rather than hard-coded).

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
        if self.adapt_b < 0:
            raise ValueError("adapt_b must be >= 0 (w is an outward/hyperpolarising current)")
        if self.adapt_tau_w <= 0:
            raise ValueError("adapt_tau_w must be > 0")
        if self.adapt_g_b < 0:
            raise ValueError("adapt_g_b must be >= 0 (g_a is an outward conductance)")
        if self.adapt_tau_a <= 0:
            raise ValueError("adapt_tau_a must be > 0")
        if self.adapt_g_b > 0:
            if self.synapse != "conductance":
                raise ValueError(
                    "adapt_g_b > 0 requires synapse='conductance': the current model has no "
                    "reversal potentials, so an adaptation conductance is meaningless there "
                    "(docs/m2d-brief.md). Use adapt_b for the M2c current-based adaptation.")
            if self.adapt_b > 0:
                raise ValueError(
                    "adapt_b (M2c, current) and adapt_g_b (M2d, conductance) are mutually "
                    "exclusive; set exactly one of them")
            if not (self.E_adapt <= self.v_reset):
                raise ValueError("need E_adapt <= v_reset (adaptation must hyperpolarise)")
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
    def decay_w(self) -> float:
        """Per-step decay of the M2c adaptation current."""
        return math.exp(-self.dt / self.adapt_tau_w)

    @property
    def adapt(self) -> bool:
        """Whether the M2c adaptation-current path runs (adapt_b = 0 keeps the pre-M2c kernel)."""
        return self.adapt_b > 0.0

    @property
    def decay_a(self) -> float:
        """Per-step decay of the M2d adaptation conductance."""
        return math.exp(-self.dt / self.adapt_tau_a)

    @property
    def avg_a(self) -> float:
        """Step-average of exp(-t/tau_a) over one step, as for avg_e / avg_i (-> 1 as dt -> 0)."""
        return self.adapt_tau_a * (1.0 - self.decay_a) / self.dt

    @property
    def adapt_g(self) -> bool:
        """Whether the M2d adaptation-conductance path runs (adapt_g_b = 0 keeps the pre-M2d kernel)."""
        return self.adapt_g_b > 0.0

    @property
    def v_lower_bound(self) -> float:
        """Lowest v the model can reach with i_ext >= 0: the most negative reversal potential
        in play. M2d's acceptance check (docs/m2d-brief.md §4) is v >= this."""
        return min(self.E_inh, self.E_adapt) if self.synapse == "conductance" else float("-inf")

    @property
    def noise_scale(self) -> float:
        """Per-step current-noise std. ASSUMPTION: scaled by sqrt(1 ms / dt) so the
        stationary voltage variance driven by the noise is independent of dt."""
        return self.noise_sigma * math.sqrt(1.0 / self.dt)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(decay_m=self.decay_m, decay_syn=self.decay_syn, c_syn=self.c_syn, ref_steps=self.ref_steps,
                 noise_scale=self.noise_scale, decay_e=self.decay_e, decay_i=self.decay_i,
                 avg_e=self.avg_e, avg_i=self.avg_i, decay_w=self.decay_w, adapt=self.adapt,
                 decay_a=self.decay_a, avg_a=self.avg_a, adapt_g=self.adapt_g,
                 v_lower_bound=self.v_lower_bound)
        return d
