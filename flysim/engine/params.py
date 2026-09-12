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


@dataclass(frozen=True)
class EngineParams:
    dt: float = 1.0            # ms. 1.0 for rate mode, 0.1 for phase-lock mode (M3).
    tau_m: float = 20.0        # ms, membrane time constant           (ASSUMPTION)
    tau_syn: float = 5.0       # ms, synaptic current time constant   (ASSUMPTION)
    t_ref: float = 2.0         # ms, absolute refractory period       (ASSUMPTION)
    v_rest: float = -65.0      # mV                                   (ASSUMPTION)
    v_reset: float = -65.0     # mV                                   (ASSUMPTION)
    v_thresh: float = -50.0    # mV                                   (ASSUMPTION)
    g: float = DEFAULT_G       # synaptic gain, mV per synaptic contact per spike (ASSUMPTION; sweep)
    noise_sigma: float = 0.0   # std of Gaussian current noise, referenced to dt = 1 ms; 0 = off
    v_floor: float | None = None  # optional hard lower clamp on v (mV); None = no clamp (default)

    def __post_init__(self) -> None:
        if self.dt <= 0 or self.tau_m <= 0 or self.tau_syn <= 0:
            raise ValueError("dt, tau_m, tau_syn must be > 0")
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
    def noise_scale(self) -> float:
        """Per-step current-noise std. ASSUMPTION: scaled by sqrt(1 ms / dt) so the
        stationary voltage variance driven by the noise is independent of dt."""
        return self.noise_sigma * math.sqrt(1.0 / self.dt)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(decay_m=self.decay_m, decay_syn=self.decay_syn, c_syn=self.c_syn, ref_steps=self.ref_steps,
                 noise_scale=self.noise_scale)
        return d
