"""Node sizing and collision-aware coordinates for the topological metro map."""

import networkx as nx
import numpy as np


def marker_sizes(values: dict[int, dict], selected: int) -> dict[int, float]:
    """Demand-dependent marker diameters (pixels), with a selected halo."""
    counts = [max(0., float(item["value"])) for item in values.values()]
    low = float(np.percentile(counts, 5)) if counts else 0.
    high = float(np.percentile(counts, 95)) if counts else 1.
    spread = max(high - low, 1.)
    return {int(station): float(10 + 14 * np.sqrt(np.clip((item["value"] - low) / spread, 0., 1.)))
            for station, item in values.items()}


def separated_layout(graph: nx.Graph, sizes: dict[int, float], selected: int,
                     width: int = 1350, height: int = 710) -> dict[int, tuple[float, float]]:
    """Constrain nodes to nonoverlapping pixel-space circles at design viewport.

    This deliberately adjusts a deterministic graph layout instead of claiming
    geographical station positions. Zooming out may visually compress markers.
    """
    original = nx.spring_layout(graph, seed=42, iterations=300, k=0.58)
    ids = list(graph.nodes)
    xy = np.array([original[node] for node in ids], dtype=float)
    xy[:, 0] = (xy[:, 0] - xy[:, 0].min()) / max(np.ptp(xy[:, 0]), 1e-9) * (width - 170) + 85
    xy[:, 1] = (xy[:, 1] - xy[:, 1].min()) / max(np.ptp(xy[:, 1]), 1e-9) * (height - 150) + 75
    anchor = xy.copy()
    radii = np.array([max(sizes.get(int(node), 10.) / 2, 21 if int(node) == selected else 0)
                      for node in ids])
    for _ in range(240):
        shift = np.zeros_like(xy)
        overlaps = 0
        for i in range(len(ids)):
            delta = xy[i] - xy[i+1:]
            distances = np.linalg.norm(delta, axis=1)
            minimums = radii[i] + radii[i+1:] + 11
            hit = np.flatnonzero(distances < minimums)
            for offset in hit:
                j = i + offset + 1
                if distances[offset] < 1e-8:
                    unit = np.array([1., 0.]) if (i+j)%2 else np.array([0., 1.])
                else:
                    unit = delta[offset] / distances[offset]
                correction = (minimums[offset] - distances[offset]) / 2 + .2
                shift[i] += unit * correction
                shift[j] -= unit * correction
                overlaps += 1
        if overlaps == 0:
            break
        xy += shift * .75
        xy += (anchor - xy) * .004
        xy[:, 0] = np.clip(xy[:, 0], 40, width - 40)
        xy[:, 1] = np.clip(xy[:, 1], 40, height - 40)
    return {int(node): tuple(map(float, xy[k])) for k, node in enumerate(ids)}


def no_collisions(layout, sizes, selected, padding=5):
    """Diagnostic at design viewport, counting marker overlaps."""
    ids = list(layout)
    for i, a in enumerate(ids):
        for b in ids[i+1:]:
            ra = max(sizes.get(a, 10)/2, 21 if a == selected else 0)
            rb = max(sizes.get(b, 10)/2, 21 if b == selected else 0)
            if np.linalg.norm(np.subtract(layout[a], layout[b])) < ra + rb + padding:
                return False
    return True
