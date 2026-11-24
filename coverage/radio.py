"""Radio geometry: horizon, line of sight over terrain, viewshed and a link budget.

Everything here is textbook VHF/UHF planning:

* The atmosphere bends radio waves slightly towards the ground. The usual way to
  account for it is to pretend the Earth is larger: radius k * R with k = 4/3.
* Over a smooth Earth an antenna h metres up reaches the horizon at sqrt(2 k R h).
  With k = 4/3 that is about 4.12 * sqrt(h) kilometres.
* Over real terrain a path is clear when the straight line between the antennas
  stays above the ground, after the ground has been raised by the Earth's bulge.
* A stricter test asks for 60 per cent of the first Fresnel zone to be clear too.

None of this is specific to any network. It answers "what is physically possible".
"""

from __future__ import annotations

import math

import numpy as np

from .terrain import EARTH_RADIUS_M, Terrain, destination, distance_m

K_FACTOR = 4.0 / 3.0
SPEED_OF_LIGHT = 299_792_458.0


def radio_horizon_m(height_m: float, k: float = K_FACTOR) -> float:
    """Distance to the radio horizon over a smooth Earth, in metres."""
    return math.sqrt(2.0 * k * EARTH_RADIUS_M * height_m)


def smooth_earth_range_m(h1_m: float, h2_m: float, k: float = K_FACTOR) -> float:
    """Longest clear path between two antennas over a smooth Earth, in metres."""
    return radio_horizon_m(h1_m, k) + radio_horizon_m(h2_m, k)


def earth_bulge_m(d1_m, d2_m, k: float = K_FACTOR):
    """How far the Earth rises above the straight chord, d1 from one end and d2 from the other."""
    return np.asarray(d1_m) * np.asarray(d2_m) / (2.0 * k * EARTH_RADIUS_M)


def fresnel_radius_m(d1_m, d2_m, frequency_hz: float):
    """Radius of the first Fresnel zone at a point d1 from one end and d2 from the other."""
    wavelength = SPEED_OF_LIGHT / frequency_hz
    d1 = np.asarray(d1_m, dtype=np.float64)
    d2 = np.asarray(d2_m, dtype=np.float64)
    total = np.maximum(d1 + d2, 1e-9)
    return np.sqrt(wavelength * d1 * d2 / total)


def free_space_loss_db(distance_m_: float, frequency_hz: float) -> float:
    """Free-space path loss in dB."""
    return 20 * math.log10(distance_m_) + 20 * math.log10(frequency_hz) + 20 * math.log10(4 * math.pi / SPEED_OF_LIGHT)


def link_margin_db(distance_m_: float, frequency_hz: float, tx_power_w: float, tx_gain_dbi: float, rx_gain_dbi: float, rx_sensitivity_dbm: float) -> float:
    """Received power minus receiver sensitivity on a clear path, in dB."""
    tx_dbm = 10 * math.log10(tx_power_w * 1000.0)
    received = tx_dbm + tx_gain_dbi + rx_gain_dbi - free_space_loss_db(distance_m_, frequency_hz)
    return received - rx_sensitivity_dbm


def profile(terrain: Terrain, lat1: float, lon1: float, lat2: float, lon2: float, samples: int = 512):
    """Terrain along the straight path between two points.

    Returns (distance_m, ground_m) arrays of length `samples`, ends included.
    """
    f = np.linspace(0.0, 1.0, samples)
    lat = lat1 + (lat2 - lat1) * f
    lon = lon1 + (lon2 - lon1) * f
    total = float(distance_m(lat1, lon1, lat2, lon2))
    return f * total, terrain.elevation(lat, lon)


def clearance_m(terrain: Terrain, site, targets_lat, targets_lon, site_height_m: float, target_height_m: float, samples: int = 768, k: float = K_FACTOR, fresnel_hz: float | None = None, fresnel_fraction: float = 0.6):
    """Smallest gap between the sight line and the ground, for many targets at once.

    `site` is (lat, lon). The line runs from `site_height_m` above the ground at the site
    to `target_height_m` above the ground at each target. The ground is raised by the
    Earth's bulge. With `fresnel_hz` set, `fresnel_fraction` of the first Fresnel zone
    radius is subtracted as well. A result >= 0 means the path is clear.
    Targets outside the terrain give -inf.
    """
    lat0, lon0 = site
    t_lat = np.atleast_1d(np.asarray(targets_lat, dtype=np.float64))
    t_lon = np.atleast_1d(np.asarray(targets_lon, dtype=np.float64))
    f = np.linspace(0.0, 1.0, samples)[None, :]
    lat = lat0 + (t_lat[:, None] - lat0) * f
    lon = lon0 + (t_lon[:, None] - lon0) * f
    ground = terrain.elevation(lat, lon)
    total = distance_m(lat0, lon0, t_lat, t_lon)[:, None]
    d1 = f * total
    d2 = total - d1
    z0 = ground[:, :1] + site_height_m
    z1 = ground[:, -1:] + target_height_m
    line = z0 + (z1 - z0) * f
    obstacle = ground + earth_bulge_m(d1, d2, k)
    if fresnel_hz is not None:
        obstacle = obstacle + fresnel_fraction * fresnel_radius_m(d1, d2, fresnel_hz)
    gap = (line - obstacle)[:, 1:-1]  # the two ends are the antennas themselves
    result = np.min(gap, axis=1)
    return np.where(np.isnan(result), -np.inf, result)


def visible(terrain: Terrain, site, targets_lat, targets_lon, site_height_m: float, target_height_m: float, **options) -> np.ndarray:
    """True for each target that has a clear path to the site."""
    return clearance_m(terrain, site, targets_lat, targets_lon, site_height_m, target_height_m, **options) >= 0.0


def viewshed(terrain: Terrain, site, site_height_m: float, target_height_m: float, radius_m: float = 150_000.0, azimuths: int = 2880, step_m: float = 90.0, k: float = K_FACTOR):
    """Everything a site can see, by sweeping rays outwards.

    Along each ray the steepest angle to the ground so far is remembered. A receiver
    `target_height_m` above the ground is visible when the angle to it is at least
    that steep. Returns a dict with the polar grid and the visibility mask:

      bearing_deg (A,), distance_m (S,), lat (A, S), lon (A, S), visible (A, S) bool
    """
    lat0, lon0 = site
    bearing = np.arange(azimuths) * (360.0 / azimuths)
    dist = np.arange(1, int(radius_m / step_m) + 1) * step_m
    lat, lon = destination(lat0, lon0, bearing[:, None], dist[None, :])
    ground = terrain.elevation(lat, lon)
    inside = ~np.isnan(ground)
    ground = np.where(inside, ground, -1e9)
    z0 = float(terrain.elevation(lat0, lon0)) + site_height_m
    drop = dist[None, :] ** 2 / (2.0 * k * EARTH_RADIUS_M)
    angle_ground = (ground - drop - z0) / dist[None, :]
    angle_target = (ground + target_height_m - drop - z0) / dist[None, :]
    blocking = np.maximum.accumulate(angle_ground, axis=1)
    before = np.concatenate([np.full((azimuths, 1), -np.inf), blocking[:, :-1]], axis=1)
    return {
        "bearing_deg": bearing,
        "distance_m": dist,
        "lat": lat,
        "lon": lon,
        "visible": (angle_target >= before) & inside,
        "inside": inside,
    }


def viewshed_area_km2(shed) -> float:
    """Visible ground area in square kilometres, from the polar grid."""
    dist = shed["distance_m"]
    step = dist[1] - dist[0]
    dtheta = 2 * math.pi / len(shed["bearing_deg"])
    cell = dist * dtheta * step  # area of one polar cell at each distance
    return float(np.sum(shed["visible"] * cell[None, :]) / 1e6)


def viewshed_reach_km(shed, percentile: float = 100.0) -> float:
    """Distance to the farthest visible ground (or a percentile of visible cells), in km."""
    mask = shed["visible"]
    if not mask.any():
        return 0.0
    dist = np.broadcast_to(shed["distance_m"][None, :], mask.shape)[mask]
    return float(np.percentile(dist, percentile) / 1000.0)
