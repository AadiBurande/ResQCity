# Erlang-C fleet sizing

## Purpose

This module provides an analytical M/M/c queueing benchmark for fleet sizing and simulator
validation. It estimates waiting probability and queue wait from the number of servers, arrival
rate, and mean service time. It does not model travel, dispatch policies, or on-scene processes.

## Assumptions

- Arrivals follow a Poisson process with rate \(\lambda\).
- Service times are exponentially distributed with rate \(\mu\) per server.
- There are \(c\) identical servers and a single queue.
- Offered load is \(a = \lambda/\mu\) Erlangs; utilisation is
  \(\rho = a/c\).
- The queue is stable only when \(\rho < 1\).

These are analytical assumptions, not claims that real emergency-service workloads follow an
M/M/c process.

## Formulation

The Erlang-C probability that an arrival must wait is

\[
P(\text{wait}) =
\frac{\frac{a^c}{c!(1-\rho)}}
{\sum_{k=0}^{c-1}\frac{a^k}{k!}+
 \frac{a^c}{c!(1-\rho)}}.
\]

The expected queueing wait is

\[
E[W_q] = \frac{P(\text{wait})}{c\mu-\lambda}
       = \frac{P(\text{wait})\,s}{c(1-\rho)},
\]

where \(s=1/\mu\) is the mean service time in seconds. For an unstable system
(\(\rho \ge 1\)), the expected queue wait is unbounded.

The implementation in `resqcity.optimization.placement` computes \(P(\text{wait})\) using a
numerically stable Erlang-B recurrence and converts it to Erlang-C. `erlang_metrics` reports
server count, offered load, utilisation, waiting probability, expected wait, and stability.

## Simulator validation

`resqcity.sim.validate_queue` strips the simulator down to an M/M/c workload and compares its
utilisation, probability of waiting, and expected queue wait against these analytical values.
The committed synthetic validation is in `results/sim/queue_validation.json`. Its recorded
worst absolute differences are:

- Utilisation: 0.000626
- \(P(\text{wait})\): 0.004362
- Expected queue wait: 1.054 seconds

The validation passed the tolerances recorded in that result file. It checks the simulated
clock, queue discipline, and busy-time accounting under the tested setup; it does not validate
travel times, hospital processes, or dispatch-policy outcomes.

## Limitations

Fleet-sizing estimates depend on the M/M/c assumptions and input rates. Emergency calls may be
time-varying, spatially heterogeneous, and correlated, and units may have different service
times. Treat the results as an analytical cross-check and sensitivity tool, not as verified
operational fleet recommendations.