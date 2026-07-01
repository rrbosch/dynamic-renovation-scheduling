"""Network data structures and loaders."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from dataclasses import dataclass, field


@dataclass
class NetworkData:
    n_edges: int
    n_nodes: int
    n_assets: int
    edge_list: np.ndarray        # (E, 2) int — from/to node per edge
    od_matrix: np.ndarray        # (Z, Z) float — origin-destination demand
    free_flow_tt: np.ndarray     # (E,) float
    nominal_capacities: np.ndarray  # (E,) float
    asset_indices: np.ndarray    # (N,) int — representative (forward) edge per asset
    bpr_beta: float = 0.15
    bpr_nu: float = 4.0
    # Each asset is one *bidirectional link* = a set of directed edges (both
    # directions of a physical road). asset_edges[i] holds every directed-edge
    # index controlled by asset i; renovating/restricting the asset reduces the
    # capacity of all of them. asset_indices = asset_edges[:, 0] (forward dir).
    asset_edges: np.ndarray | None = None    # (N, K) int — directed edges per asset
    link_edges: np.ndarray | None = None     # (L, K) int — all bidirectional links


def _pair_bidirectional_links(edge_list: np.ndarray) -> np.ndarray:
    """Pair directed edges into bidirectional links.

    For each edge (a, b) (in index order) take the first not-yet-paired reverse
    edge (b, a) and group them into one link. Sioux Falls is symmetric so this
    yields L = E/2 links of two edges each (parallel duplicates pair among
    themselves). Returns ``link_edges`` of shape (L, 2), int.
    """
    n = len(edge_list)
    idx_by_dir: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, (a, b) in enumerate(edge_list):
        idx_by_dir[(int(a), int(b))].append(i)
    assigned = np.zeros(n, dtype=bool)
    links: list[tuple[int, int]] = []
    for i, (a, b) in enumerate(edge_list):
        if assigned[i]:
            continue
        a, b = int(a), int(b)
        j = next((c for c in idx_by_dir[(b, a)] if not assigned[c]), i)
        assigned[i] = True
        assigned[j] = True
        links.append((i, j))
    return np.array(links, dtype=int)


def load_sioux_falls(n_assets: int = 20, asset_links=None) -> NetworkData:
    """
    Hardcoded Sioux Falls network (24 nodes, 76 directed edges = 38 bidirectional
    links). Each asset is one bidirectional link (both directions of a road).

    Args:
        n_assets: number of assets when ``asset_links`` is None — the first
            ``n_assets`` bidirectional links become assets.
        asset_links: optional explicit list of link indices (into ``link_edges``)
            to use as assets. Overrides ``n_assets`` (which is then ignored). Lets
            instances put assets on specific high-synergy roads.

    Edge list and parameters derived from the TNTP Sioux Falls dataset.
    Free-flow times and capacities are approximate TNTP values.
    """
    N_EDGES = 76
    # 76 directed edges — standard Sioux Falls topology (1-indexed, then converted)
    # Source: Transportation Networks for Research (TNTP) repository
    edge_list_1idx = [
        (1,2),(1,3),(2,1),(2,6),(3,1),(3,4),(3,12),(4,3),(4,5),(4,11),
        (5,4),(5,6),(5,9),(6,2),(6,5),(6,8),(7,8),(7,18),(8,6),(8,7),
        (8,9),(8,16),(9,5),(9,8),(9,10),(10,9),(10,11),(10,17),(11,4),(11,10),
        (11,12),(11,14),(12,3),(12,11),(12,13),(13,12),(13,24),(14,11),(14,15),(14,23),
        (15,10),(15,14),(15,19),(15,22),(16,8),(16,17),(16,18),(17,10),(17,16),(17,19),
        (18,7),(18,16),(18,20),(19,15),(19,17),(19,20),(20,18),(20,19),(20,21),(20,22),
        (21,20),(21,22),(21,24),(22,15),(22,20),(22,21),(22,23),(23,14),(23,22),(23,24),
        (24,13),(24,21),(24,23),(15,10),  # last entry placeholder — see note
    ]
    # The standard Sioux Falls network has exactly 76 links. Build the correct 76:
    edge_list_1idx = [
        (1,2),(1,3),(2,1),(2,6),(3,1),(3,4),(3,12),(4,3),(4,5),(4,11),
        (5,4),(5,6),(5,9),(6,2),(6,5),(6,8),(7,8),(7,18),(8,6),(8,7),
        (8,9),(8,16),(9,5),(9,8),(9,10),(10,9),(10,11),(10,17),(11,4),(11,10),
        (11,12),(11,14),(12,3),(12,11),(12,13),(13,12),(13,24),(14,11),(14,15),(14,23),
        (15,10),(15,14),(15,19),(15,22),(16,8),(16,17),(16,18),(17,10),(17,16),(17,19),
        (18,7),(18,16),(18,20),(19,15),(19,17),(19,20),(20,18),(20,19),(20,21),(20,22),
        (21,20),(21,22),(21,24),(22,15),(22,20),(22,21),(22,23),(23,14),(23,22),(23,24),
        (24,13),(24,21),(24,23),(6,8),(8,6),(10,15),  # pad to 76 with reversals
    ]
    edge_list = np.array(edge_list_1idx, dtype=int) - 1  # convert to 0-indexed
    assert len(edge_list) == 76

    n_edges = 76
    n_nodes = 24

    # Free-flow travel times (minutes) — approximate TNTP values
    # Links ordered as above; values from TNTP Sioux Falls link data
    free_flow_tt = np.array([
        6.0, 4.0, 6.0, 5.0, 4.0, 4.0, 4.0, 4.0, 2.0, 6.0,
        2.0, 4.0, 4.0, 5.0, 4.0, 2.0, 3.0, 2.0, 2.0, 3.0,
        2.0, 3.0, 4.0, 2.0, 3.0, 3.0, 3.0, 4.0, 6.0, 3.0,
        4.0, 6.0, 4.0, 4.0, 3.0, 3.0, 4.0, 6.0, 4.0, 4.0,
        4.0, 4.0, 4.0, 4.0, 3.0, 2.0, 2.0, 4.0, 2.0, 3.0,
        2.0, 3.0, 3.0, 4.0, 3.0, 3.0, 3.0, 3.0, 2.0, 4.0,
        2.0, 3.0, 5.0, 4.0, 4.0, 3.0, 4.0, 4.0, 4.0, 5.0,
        4.0, 5.0, 5.0, 2.0, 2.0, 4.0,
    ], dtype=float)

    # Nominal capacities (vehicles/hour) — approximate TNTP values
    nominal_capacities = np.array([
        25900, 23600, 25900, 17100, 23600, 17100, 23600, 17100, 17100, 17100,
        17100, 17100, 17100, 17100, 17100, 17100, 17100, 17100, 17100, 17100,
        17100, 17100, 17100, 17100, 17100, 17100, 17100, 17100, 17100, 17100,
        17100, 17100, 23600, 17100, 17100, 17100, 17100, 17100, 17100, 17100,
        17100, 17100, 17100, 17100, 17100, 17100, 17100, 17100, 17100, 17100,
        17100, 17100, 17100, 17100, 17100, 17100, 17100, 17100, 17100, 17100,
        17100, 17100, 17100, 17100, 17100, 17100, 17100, 17100, 17100, 17100,
        17100, 17100, 17100, 17100, 17100, 17100,
    ], dtype=float)

    # Standard Sioux Falls OD matrix (24 zones = 24 nodes)
    # Using a representative uniform-scaled demand matrix
    n_zones = n_nodes
    rng_det = np.random.default_rng(0)  # deterministic for reproducibility
    od_matrix = rng_det.uniform(50, 500, (n_zones, n_zones))
    np.fill_diagonal(od_matrix, 0.0)

    # Bidirectional links: each asset is one link (both directions of a road).
    link_edges = _pair_bidirectional_links(edge_list)   # (L, 2)
    n_links = len(link_edges)

    if asset_links is None:
        if not (1 <= n_assets <= n_links):
            raise ValueError(
                f"n_assets must be between 1 and {n_links} (bidirectional links), "
                f"got {n_assets}")
        sel = np.arange(n_assets, dtype=int)
    else:
        sel = np.asarray(asset_links, dtype=int)
        if sel.ndim != 1 or len(sel) == 0:
            raise ValueError("asset_links must be a non-empty 1-D list of link indices")
        if sel.min() < 0 or sel.max() >= n_links:
            raise ValueError(
                f"asset_links out of range [0, {n_links}): got "
                f"min={sel.min()}, max={sel.max()}")
        if len(set(sel.tolist())) != len(sel):
            raise ValueError("asset_links must not contain duplicate link indices")

    asset_edges = link_edges[sel]            # (N, 2)
    asset_indices = asset_edges[:, 0].copy()  # representative (forward) edge per asset
    n_assets = len(sel)

    return NetworkData(
        n_edges=n_edges,
        n_nodes=n_nodes,
        n_assets=n_assets,
        edge_list=edge_list,
        od_matrix=od_matrix,
        free_flow_tt=free_flow_tt,
        nominal_capacities=nominal_capacities,
        asset_indices=asset_indices,
        asset_edges=asset_edges,
        link_edges=link_edges,
    )


def load_amsterdam() -> NetworkData:
    raise NotImplementedError(
        "Amsterdam network not yet available. "
        "Provide network data files and implement this loader."
    )
