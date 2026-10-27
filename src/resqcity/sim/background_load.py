"""Out-of-scope workload that makes units genuinely busy (phase 7 slice 4; D-031 item 3, L30).

**Why this exists.** Phase 6 measured the modelled city at 0.26-0.32 % utilisation: the fleet
serves each agency's entire workload, while the incident generator models only the multi-agency
subset in scope for this study. Against the one verified volume figure the project has -- 23,887
Dial-112 calls in Pune in July 2026, i.e. 771/day -- the modelled police workload is about 1/138
of the real one. With every unit always free, travel time is the only cost, the nearest unit is
always available, and joint dispatch would beat sequential dispatch for a reason that has nothing
to do with coordination. So this process is part of the model, not a refinement of it.

**What it does.** Per agency, jobs arrive as a Poisson process and each occupies one unit for an
exponentially distributed time. The arrival rate is derived from the target utilisation rather
than configured directly:

    lambda_a = rho_a * c_a / S_a

for ``c_a`` units and mean service time ``S_a``, which gives an offered load of ``rho_a * c_a``
erlangs over ``c_a`` servers. ``S_a`` comes from ``configs/optimization.yaml`` ->
``erlang.service_time_s`` so that this process, the Erlang-C cross-check of phase 6 and the
validation of slice 7 cannot drift apart.

**Queueing is real.** A job that finds no free unit waits in a FIFO queue and is started by the
next unit of its agency to become free. That is what makes slice 7's M/M/c comparison meaningful:
if the queue were dropped, waiting time would be zero by construction and there would be nothing
to validate. Jobs are never pre-empted for an in-scope incident
(``configs/sim.yaml`` -> ``background.preempt``), which is pessimistic -- a real control room
would recall a patrol for a serious crash -- and is a stated limitation.
"""

import logging
from collections import deque
from collections.abc import Generator
from dataclasses import dataclass, field

from resqcity.common.schemas import Agency, UnitState
from resqcity.optimization.config import OptimizationConfig
from resqcity.sim.config import SimConfig

log = logging.getLogger(__name__)


@dataclass
class BackgroundJob:
    agency: Agency
    arrived_s: float
    started_s: float | None = None
    service_s: float | None = None

    @property
    def wait_s(self) -> float | None:
        return None if self.started_s is None else self.started_s - self.arrived_s


@dataclass
class BackgroundStats:
    """Everything slice 7 needs to compare the simulator with Erlang-C."""

    arrived: int = 0
    started: int = 0
    waits_s: list[float] = field(default_factory=list)
    service_s: list[float] = field(default_factory=list)
    unfinished_at_end: int = 0

    @property
    def mean_wait_s(self) -> float:
        return sum(self.waits_s) / len(self.waits_s) if self.waits_s else 0.0

    @property
    def p_wait(self) -> float:
        """Share of jobs that had to wait at all -- Erlang-C's C(c, a)."""
        if not self.waits_s:
            return 0.0
        return sum(w > 0 for w in self.waits_s) / len(self.waits_s)


class BackgroundLoad:
    """Drives the out-of-scope workload for one replication."""

    def __init__(self, engine: object, cfg: SimConfig, opt: OptimizationConfig) -> None:
        self.engine = engine
        self.cfg, self.opt = cfg, opt
        self.queues: dict[Agency, deque[BackgroundJob]] = {a: deque() for a in Agency}
        self.stats: dict[Agency, BackgroundStats] = {a: BackgroundStats() for a in Agency}
        self.rates: dict[Agency, float] = {}
        for agency in Agency:
            self.rates[agency] = self.arrival_rate(agency)
        # Each stream is fetched ONCE and kept. Asking the engine for "the" stream per job would
        # make the draws depend on the caller caching generators by name: with a fresh generator
        # each time, every service time is the same number and utilisation lands systematically
        # below the target (0.86x, the first draw of that seed) rather than noisily around it.
        self._service_rng = {a: self.engine.rng(f"background_service/{a.value}") for a in Agency}

    # -- sizing -------------------------------------------------------------------------------

    def service_mean_s(self, agency: Agency) -> float:
        return float(self.opt.erlang.service_time_s.value[agency])

    def arrival_rate(self, agency: Agency) -> float:
        """Jobs per second, derived from the target utilisation (see the module docstring)."""
        if not self.cfg.background.enabled:
            return 0.0
        rho = float(self.cfg.background.target_utilisation.value[agency])
        servers = self.engine.fleet.count(agency)
        if servers == 0 or rho <= 0.0:
            return 0.0
        return rho * servers / self.service_mean_s(agency)

    def offered_load(self, agency: Agency) -> float:
        """Erlangs: lambda * S. Equals rho * c by construction, which slice 7 checks."""
        return self.rates[agency] * self.service_mean_s(agency)

    # -- processes ----------------------------------------------------------------------------

    def start(self) -> None:
        """Register the arrival process for each agency and the unit-freed hook."""
        if not self.cfg.background.enabled:
            return
        self.engine.on_unit_free.append(self._unit_freed)
        for agency in Agency:
            if self.rates[agency] > 0:
                self.engine.env.process(self._arrivals(agency))

    def _arrivals(self, agency: Agency) -> Generator:
        rng = self.engine.rng(f"background_arrivals/{agency.value}")
        rate = self.rates[agency]
        while True:
            yield self.engine.env.timeout(float(rng.exponential(1.0 / rate)))
            job = BackgroundJob(agency=agency, arrived_s=self.engine.now_s)
            self.stats[agency].arrived += 1
            self.queues[agency].append(job)
            self._pump(agency)

    def _unit_freed(self, index: int) -> None:
        """Called by the engine whenever a unit returns to IDLE."""
        self._pump(self.engine.fleet.units[index].agency)

    def _pump(self, agency: Agency) -> None:
        """Start as many queued jobs as there are free units of this agency."""
        while self.queues[agency]:
            free = self.engine.fleet.claimable(agency)
            # a pre-alerted unit is claimable for an INCIDENT but must not be pulled onto
            # out-of-scope work: it was held back for a specific candidate
            free = [i for i in free if self.engine.fleet.units[i].state is UnitState.IDLE]
            if not free:
                return
            job = self.queues[agency].popleft()
            # The unit is taken NOW, not inside the process: a SimPy process does not start
            # until the environment next runs, so deferring the state change would let this
            # loop hand the same unit to every queued job.
            index = free[0]
            job.started_s = self.engine.now_s
            stats = self.stats[agency]
            stats.started += 1
            stats.waits_s.append(job.wait_s or 0.0)
            draw = self._service_rng[agency]
            job.service_s = float(draw.exponential(self.service_mean_s(agency)))
            stats.service_s.append(job.service_s)
            self.engine.fleet.set_state(index, UnitState.ON_OTHER_TASK, self.engine.now_s)
            self.engine.env.process(self._serve(index, job.service_s))

    def _serve(self, index: int, service_s: float) -> Generator:
        """Hold an already-claimed unit for the service time, then release it."""
        yield self.engine.env.timeout(service_s)
        self.engine.fleet.set_state(index, UnitState.IDLE, self.engine.now_s)
        self.engine.unit_freed(index)

    # -- reporting ----------------------------------------------------------------------------

    def report(self) -> dict[str, object]:
        out: dict[str, object] = {
            "enabled": self.cfg.background.enabled,
            "service_time_source": "configs/optimization.yaml -> erlang.service_time_s",
            "by_agency": {},
        }
        for agency in Agency:
            stats = self.stats[agency]
            out["by_agency"][agency.value] = {
                "target_utilisation": float(self.cfg.background.target_utilisation.value[agency]),
                "arrival_per_day": self.rates[agency] * 86400.0,
                "service_mean_s": self.service_mean_s(agency),
                "offered_load_erlangs": self.offered_load(agency),
                "jobs_arrived": stats.arrived,
                "jobs_started": stats.started,
                "queued_at_end": len(self.queues[agency]),
                "mean_wait_s": stats.mean_wait_s,
                "p_wait": stats.p_wait,
            }
        return out
