"""Covering a road with repeaters: route sampling, candidate hilltops and a greedy cover."""

from __future__ import annotations

import json

import numpy as np

from .radio import visible
from .terrain import Terrain, distance_m


def load_route(path: str) -> np.ndarray:
    """Reads a GeoJSON LineString and returns its vertices as an (N, 2) array of (lat, lon)."""
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    feature = data["features"][0] if data.get("type") == "FeatureCollection" else data
    coords = np.asarray(feature["geometry"]["coordinates"], dtype=np.float64)
    return coords[:, ::-1].copy()


def densify(route: np.ndarray, spacing_m: float = 500.0):
    """Points every `spacing_m` along a route. Returns (points (M, 2), chainage_m (M,))."""
    seg = distance_m(route[:-1, 0], route[:-1, 1], route[1:, 0], route[1:, 1])
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    marks = np.arange(0.0, cum[-1], spacing_m)
    marks = np.append(marks, cum[-1])
    lat = np.interp(marks, cum, route[:, 0])
    lon = np.interp(marks, cum, route[:, 1])
    return np.column_stack([lat, lon]), marks


def candidate_hilltops(terrain: Terrain, route_points: np.ndarray, within_m: float = 15_000.0, block: int = 24, keep: int = 160) -> np.ndarray:
    """High points near the route, as (lat, lon, elevation_m) rows.

    The grid is cut into blocks of `block` cells (about 2 km for SRTM3). The highest cell
    of each block is a candidate if it lies within `within_m` of the route. Candidates are
    ranked by how far they stand above the average ground within about 10 km, and the best
    `keep` are returned. This is a search for plausible sites, not a claim about real ones.
    """
    z = terrain.z
    rows = (z.shape[0] // block) * block
    cols = (z.shape[1] // block) * block
    blocks = z[:rows, :cols].reshape(rows // block, block, cols // block, block).swapaxes(1, 2).reshape(rows // block, cols // block, block * block)
    arg = blocks.argmax(axis=2)
    peak = blocks.max(axis=2)
    br, bc = np.meshgrid(np.arange(rows // block), np.arange(cols // block), indexing="ij")
    r = br * block + arg // block
    c = bc * block + arg % block
    lat, lon = terrain.cell_latlon(r.ravel(), c.ravel())
    elev = peak.ravel()

    # Distance to the nearest route point, in chunks to keep memory small.
    near = np.full(lat.shape, np.inf)
    for start in range(0, len(route_points), 64):
        part = route_points[start : start + 64]
        d = distance_m(lat[:, None], lon[:, None], part[None, :, 0], part[None, :, 1])
        near = np.minimum(near, d.min(axis=1))
    close = near <= within_m

    # Height above the surroundings: the block means, smoothed over about 10 km.
    mean = blocks.mean(axis=2)
    pad = 5
    padded = np.pad(mean, pad, mode="edge")
    window = np.zeros_like(mean)
    for dr in range(-pad, pad + 1):
        for dc in range(-pad, pad + 1):
            window += padded[pad + dr : pad + dr + mean.shape[0], pad + dc : pad + dc + mean.shape[1]]
    window /= (2 * pad + 1) ** 2
    standout = (peak - window).ravel()

    index = np.flatnonzero(close)
    index = index[np.argsort(-standout[index])][:keep]
    return np.column_stack([lat[index], lon[index], elev[index]])


def coverage_matrix(terrain: Terrain, sites: np.ndarray, points: np.ndarray, site_height_m: float, target_height_m: float, max_range_m: float = 120_000.0, **options) -> np.ndarray:
    """cover[i, j] is True when site i has a clear path to route point j."""
    cover = np.zeros((len(sites), len(points)), dtype=bool)
    for i, (lat, lon, _elev) in enumerate(sites):
        d = distance_m(lat, lon, points[:, 0], points[:, 1])
        near = d <= max_range_m
        if near.any():
            cover[i, near] = visible(terrain, (lat, lon), points[near, 0], points[near, 1], site_height_m, target_height_m, **options)
    return cover


def link_matrix(terrain: Terrain, sites: np.ndarray, site_height_m: float, max_range_m: float = 150_000.0, **options) -> np.ndarray:
    """link[i, j] is True when the antennas of sites i and j see each other."""
    n = len(sites)
    link = np.zeros((n, n), dtype=bool)
    for i in range(n):
        lat, lon = sites[i, 0], sites[i, 1]
        d = distance_m(lat, lon, sites[:, 0], sites[:, 1])
        near = (d <= max_range_m) & (np.arange(n) != i)
        if near.any():
            link[i, near] = visible(terrain, (lat, lon), sites[near, 0], sites[near, 1], site_height_m, site_height_m, **options)
    return link | link.T


def greedy_cover(cover: np.ndarray, link: np.ndarray | None = None, target: float = 1.0, max_sites: int = 40):
    """Picks sites one at a time, each time the one that adds the most uncovered points.

    With `link` given, every site after the first must see at least one site already
    chosen, so the result is one connected chain of repeaters and not a set of islands.
    Returns (chosen site indices, covered fraction after each choice).
    Greedy set cover is not guaranteed to be the minimum; it is a good upper bound.
    """
    points = cover.shape[1]
    covered = np.zeros(points, dtype=bool)
    chosen: list[int] = []
    history: list[float] = []
    while covered.mean() < target and len(chosen) < max_sites:
        gain = (cover & ~covered[None, :]).sum(axis=1)
        if chosen:
            gain[chosen] = 0
            if link is not None:
                reachable = link[chosen].any(axis=0)
                gain = np.where(reachable, gain, 0)
        best = int(gain.argmax())
        if gain[best] == 0:
            break
        chosen.append(best)
        covered |= cover[best]
        history.append(float(covered.mean()))
    return chosen, history
