"""Traffic Assignment Problem solvers."""
from __future__ import annotations

import numpy as np
from collections import OrderedDict
from typing import Protocol

from env.network import NetworkData


class TAPSolver(Protocol):
    def solve(self, capacities: np.ndarray) -> np.ndarray:
        """
        capacities: (E,) effective edge capacities
        returns:    (E,) equilibrium flows
        """
        ...


# ---------------------------------------------------------------------------
# BPR helper
# ---------------------------------------------------------------------------

def _bpr_times(flows: np.ndarray, capacities: np.ndarray, tt0: np.ndarray,
               beta: float, nu: float) -> np.ndarray:
    """BPR travel time function, vectorized over edges."""
    ratio = np.where(capacities > 0, flows / capacities, 0.0)
    return tt0 * (1.0 + beta * ratio ** nu)


# ---------------------------------------------------------------------------
# Bounded LRU cache
# ---------------------------------------------------------------------------

class _BoundedCache:
    """LRU-evicting cache for TAP flow arrays. Prevents unbounded memory growth."""

    def __init__(self, maxsize: int = 10_000):
        self._d: OrderedDict[bytes, np.ndarray] = OrderedDict()
        self._maxsize = maxsize

    def get(self, key: bytes) -> np.ndarray | None:
        if key in self._d:
            self._d.move_to_end(key)
            return self._d[key]
        return None

    def put(self, key: bytes, value: np.ndarray) -> None:
        if key in self._d:
            self._d.move_to_end(key)
        else:
            self._d[key] = value
            if len(self._d) > self._maxsize:
                self._d.popitem(last=False)


# ---------------------------------------------------------------------------
# Numba import guard
# ---------------------------------------------------------------------------

try:
    from numba import njit
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):  # type: ignore[misc]
        """No-op decorator when numba is absent."""
        def decorator(fn):
            return fn
        return decorator if (args and callable(args[0])) else decorator


# ---------------------------------------------------------------------------
# JIT-compiled core functions
# ---------------------------------------------------------------------------

@njit(cache=True)
def _dijkstra_numba(source: int, row_ptr, col_idx, edge_id, costs, n_nodes: int):
    """
    O(V²) scan-based Dijkstra with predecessor tracking.

    Optimal for small V (e.g. Sioux Falls V=24): no heap allocation, tight
    inner loop of V comparisons compiled to native code.

    Returns pred_edge: (n_nodes,) int32, pred_edge[v] = edge index on the
    shortest path from source to v (-1 if unreachable).
    """
    dist = np.full(n_nodes, np.inf)
    pred_edge = np.full(n_nodes, -1, dtype=np.int32)
    visited = np.zeros(n_nodes, dtype=np.bool_)
    dist[source] = 0.0

    for _ in range(n_nodes):
        # Linear scan: find unvisited node with minimum distance
        u = -1
        min_d = np.inf
        for i in range(n_nodes):
            if not visited[i] and dist[i] < min_d:
                min_d = dist[i]
                u = i
        if u == -1:
            break
        visited[u] = True

        # Relax outgoing edges
        for k in range(row_ptr[u], row_ptr[u + 1]):
            v = col_idx[k]
            eid = edge_id[k]
            nd = dist[u] + costs[eid]
            if nd < dist[v]:
                dist[v] = nd
                pred_edge[v] = eid

    return pred_edge


@njit(cache=True)
def _fw_line_search(flows, direction, capacities, tt0, beta, nu, n_iter=30):
    """
    Frank-Wolfe line search via bisection on dB/dα = Σ tt_e(f+α·d)·d_e = 0.
    dB/dα is strictly increasing (Beckmann is convex), so bisection finds the
    exact root.  30 iterations → precision ~1e-9, far below FW gap tolerance.
    """
    n = len(flows)

    # Evaluate derivative at α=0 (flows unchanged)
    d_lo = 0.0
    for e in range(n):
        c_e = capacities[e]
        ratio = flows[e] / c_e if c_e > 0.0 else 0.0
        d_lo += tt0[e] * (1.0 + beta * ratio ** nu) * direction[e]
    if d_lo >= 0.0:
        return 0.0

    # Evaluate derivative at α=1 (full step to AoN direction)
    d_hi = 0.0
    for e in range(n):
        f_e = flows[e] + direction[e]
        if f_e < 0.0:
            f_e = 0.0
        c_e = capacities[e]
        ratio = f_e / c_e if c_e > 0.0 else 0.0
        d_hi += tt0[e] * (1.0 + beta * ratio ** nu) * direction[e]
    if d_hi <= 0.0:
        return 1.0

    # Bisect: find α* where derivative = 0
    lo, hi = 0.0, 1.0
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        d_mid = 0.0
        for e in range(n):
            f_e = flows[e] + mid * direction[e]
            if f_e < 0.0:
                f_e = 0.0
            c_e = capacities[e]
            ratio = f_e / c_e if c_e > 0.0 else 0.0
            d_mid += tt0[e] * (1.0 + beta * ratio ** nu) * direction[e]
        if d_mid < 0.0:
            lo = mid
        else:
            hi = mid

    return 0.5 * (lo + hi)


@njit(cache=True)
def _aon_numba(od_matrix, row_ptr, col_idx, edge_id, costs,
               n_nodes: int, n_zones: int, n_edges: int, edge_tails):
    """
    All-or-nothing assignment, fully JIT-compiled.

    Runs one Dijkstra per origin zone, then traces each OD path to accumulate
    flows. Replaces the pure-Python _all_or_nothing + _dijkstra_pred combo.
    """
    flows = np.zeros(n_edges)

    for orig in range(n_zones):
        # Skip origins with no outgoing demand
        row_sum = 0.0
        for d in range(n_zones):
            row_sum += od_matrix[orig, d]
        if row_sum <= 0.0:
            continue

        pred_edge = _dijkstra_numba(orig, row_ptr, col_idx, edge_id, costs, n_nodes)

        for dest in range(n_zones):
            demand = od_matrix[orig, dest]
            if demand <= 0.0 or dest == orig:
                continue
            # Trace path: dest → orig via predecessor edges
            node = dest
            while pred_edge[node] != -1:
                eid = pred_edge[node]
                flows[eid] += demand
                node = edge_tails[eid]
                if node == orig:
                    break

    return flows


# ---------------------------------------------------------------------------
# FastTAP
# ---------------------------------------------------------------------------

class FastTAP:
    """
    Frank-Wolfe TAP solver with numba-JIT compiled all-or-nothing assignment.

    Builds a CSR graph representation once on init; each solve calls the
    fully-compiled _aon_numba function instead of the pure-Python Dijkstra.
    Frank-Wolfe loop and line search remain in Python (numpy, already fast).

    Expected speedup: 10–30× over a pure-Python FW solver on Sioux Falls (V=24, E=76).
    """

    def __init__(self, network: NetworkData, rel_gap_tol: float = 1e-3):
        if not _NUMBA_AVAILABLE:
            raise ImportError(
                "FastTAP requires numba. Install with: pip install numba"
            )

        self.network = network
        self.rel_gap_tol = rel_gap_tol
        self._last_flows: np.ndarray | None = None
        self._cache = _BoundedCache(10_000)

        # Build CSR representation from edge_list
        el = network.edge_list          # (E, 2) int, column 0 = tail, column 1 = head
        n_nodes = network.n_nodes
        n_edges = network.n_edges

        tails = el[:, 0].astype(np.int32)
        heads = el[:, 1].astype(np.int32)

        degree = np.bincount(tails, minlength=n_nodes)
        row_ptr = np.zeros(n_nodes + 1, dtype=np.int32)
        np.cumsum(degree, out=row_ptr[1:])

        pos = row_ptr[:-1].copy()
        col_idx = np.empty(n_edges, dtype=np.int32)
        edge_id = np.empty(n_edges, dtype=np.int32)
        for eid in range(n_edges):
            u = int(tails[eid])
            slot = int(pos[u])
            col_idx[slot] = heads[eid]
            edge_id[slot] = eid
            pos[u] += 1

        self._row_ptr = row_ptr
        self._col_idx = col_idx
        self._edge_id = edge_id
        self._edge_tails = tails.copy()

        # JIT warmup: trigger numba compilation now so first real solve is fast.
        # Zero OD matrix → all origins skipped immediately → compiles in µs.
        _dummy_od = np.zeros((n_nodes, n_nodes), dtype=np.float64)
        _dummy_costs = np.ones(n_edges, dtype=np.float64)
        _aon_numba(_dummy_od, self._row_ptr, self._col_idx, self._edge_id,
                   _dummy_costs, n_nodes, n_nodes, n_edges, self._edge_tails)
        _fw_line_search(
            np.zeros(n_edges), np.zeros(n_edges),
            np.ones(n_edges), np.ones(n_edges),
            0.15, 4.0,
        )

    def solve(self, capacities: np.ndarray) -> np.ndarray:
        key = capacities.tobytes()
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        net = self.network
        tt0 = net.free_flow_tt
        od = net.od_matrix
        beta = net.bpr_beta
        nu = net.bpr_nu
        n_edges = net.n_edges
        n_nodes = net.n_nodes
        n_zones = od.shape[0]

        rp = self._row_ptr
        ci = self._col_idx
        ei = self._edge_id
        et = self._edge_tails

        # Warm start or free-flow AoN
        if self._last_flows is not None:
            flows = self._last_flows.copy()
        else:
            flows = _aon_numba(od, rp, ci, ei, tt0, n_nodes, n_zones, n_edges, et)

        tt = _bpr_times(flows, capacities, tt0, beta, nu)

        max_iter = 100
        rel_gap_tol = self.rel_gap_tol

        for _ in range(max_iter):
            aon_flows = _aon_numba(od, rp, ci, ei, tt, n_nodes, n_zones, n_edges, et)

            total_demand = od.sum()
            if total_demand > 0:
                sptt = np.dot(tt, aon_flows)
                current_tt_cost = np.dot(tt, flows)
                rel_gap = (current_tt_cost - sptt) / (abs(sptt) + 1e-12)
                if rel_gap < rel_gap_tol:
                    break

            direction = aon_flows - flows

            alpha = _fw_line_search(flows, direction, capacities, tt0, beta, nu)
            flows = np.maximum(flows + alpha * direction, 0.0)
            tt = _bpr_times(flows, capacities, tt0, beta, nu)

        self._last_flows = flows.copy()
        self._cache.put(key, flows.copy())
        return flows


# ---------------------------------------------------------------------------
# NullTAP — zero-flow solver for curriculum pre-training
# ---------------------------------------------------------------------------

class NullTAP:
    """Zero-flow TAP. Returns all-zero edge flows.
    Used with traffic_cost_factor=0 so c_travel is always 0.
    Baseline = 0, step flows = 0 → extra_veh_hours = 0 → c_travel = 0.
    """
    def solve(self, capacities: np.ndarray) -> np.ndarray:
        return np.zeros_like(capacities)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_tap(network: NetworkData, backend: str = 'fast') -> TAPSolver:
    """
    Returns the requested TAP backend.

    backend='fast'   FastTAP — numba JIT-compiled Frank-Wolfe (default).
    backend='null'   NullTAP — zero flows, for curriculum pre-training.
    """
    if backend == 'fast':
        return FastTAP(network)
    if backend == 'null':
        return NullTAP()
    raise ValueError(
        f"Unknown TAP backend: {backend!r}. Valid options: 'fast', 'null'."
    )
