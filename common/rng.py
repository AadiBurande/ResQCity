"""Named random-number streams for reproducible experiments with common random numbers.

Every stochastic process family (incident arrivals, calls, detection, ...) draws from its own
stream, seeded from ``(seed_base, replication, crc32(name), *path)``. Replication ``r`` therefore
sees the same random inputs under every policy, and adding a new stream never shifts existing
ones. The optional integer ``path`` (e.g. hour index, incident index) gives sub-streams, so changing
a parameter that affects one draw (say, an injury probability) does not reshuffle unrelated draws
(arrival times, locations) in a sensitivity sweep.
"""

import zlib

import numpy as np


def stream_key(name: str) -> int:
    """Stable 32-bit key for a stream name (Python's ``hash`` is salted per process)."""
    return zlib.crc32(name.encode("utf-8"))


def stream(seed_base: int, replication: int, name: str, *path: int) -> np.random.Generator:
    if seed_base < 0 or replication < 0 or any(p < 0 for p in path):
        raise ValueError("seed_base, replication and path entries must be non-negative")
    entropy = [seed_base, replication, stream_key(name), *path]
    return np.random.default_rng(np.random.SeedSequence(entropy))
