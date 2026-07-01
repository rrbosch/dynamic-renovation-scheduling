"""Stateless, counter/hash-based random number generation.

A single ``keyed_philox`` primitive turns an arbitrary key — built from a
string *phase* tag plus integers / bytes / arrays — into a fresh, deterministic
``numpy`` Philox generator. This is the foundation for reproducible, parallelism-
invariant randomness throughout the project:

  * Environment transitions and initial-condition sampling key on
    ``(phase, seed, episode_idx)`` (see ``env.mdp.InfraEnv.begin_episode``).
  * Rollout agents key on ``("rollout", seed, root_state, decision_t, rollout_idx)``
    (see ``agents.rollout.rollout_noise``).

Because the *phase* tag is folded into the hash, different phases
("training" / "evaluation" / "rollout" / ...) produce structurally independent
streams — so an agent's internal rollout simulations can never coincide with the
real evaluation transitions — while the same key reproduces identical draws
(common random numbers). No mutable global/instance RNG state is involved.
"""
from __future__ import annotations

import hashlib

import numpy as np


def keyed_philox(*key_parts) -> np.random.Generator:
    """Return a deterministic ``Philox`` generator seeded from an arbitrary key.

    ``key_parts`` may mix ``str`` (e.g. a phase tag), ``bytes`` (e.g.
    ``state.features().tobytes()``), and integers / array-likes (seed, episode
    index, timestep, ...). The parts are hashed with blake2b into a 128-bit
    digest used to seed Philox. Distinct key tuples give independent streams;
    identical key tuples give identical streams (common random numbers).

    Type tags ('s'/'b'/'i') and a separator are mixed in so that, e.g.,
    ``("ab", 1)`` and ``("a", "b1")`` cannot collide.
    """
    h = hashlib.blake2b(digest_size=16)
    for p in key_parts:
        if isinstance(p, str):
            h.update(b"s")
            h.update(p.encode("utf-8"))
            h.update(b"\x00")
        elif isinstance(p, (bytes, bytearray)):
            h.update(b"b")
            h.update(bytes(p))
            h.update(b"\x00")
        else:
            h.update(b"i")
            h.update(np.ascontiguousarray(np.asarray(p, dtype=np.int64)).tobytes())
            h.update(b"\x00")
    seed_int = int.from_bytes(h.digest(), "big")
    return np.random.Generator(np.random.Philox(np.random.SeedSequence(seed_int)))
