"""SRTM elevation tiles: loading, mosaicking and sampling.

An SRTM3 tile (.hgt) is a square of 1201 x 1201 big-endian signed 16-bit integers,
covering one degree by one degree at three arc-seconds (about 90 m). The file name
gives the south-west corner, the first row is the northern edge, and neighbouring
tiles share their edge row and column. Voids are stored as -32768.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass

import numpy as np

SAMPLES = 1201  # samples per tile edge for SRTM3
VOID = -32768
EARTH_RADIUS_M = 6_371_000.0

_NAME = re.compile(r"^([NS])(\d{2})([EW])(\d{3})\.hgt$", re.IGNORECASE)


def tile_name(lat: int, lon: int) -> str:
    """File name of the tile whose south-west corner is (lat, lon), e.g. N25W101.hgt."""
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    return f"{ns}{abs(lat):02d}{ew}{abs(lon):03d}.hgt"


def read_tile(path: str) -> np.ndarray:
    """Reads one SRTM3 tile as a (1201, 1201) int16 array, row 0 = northern edge."""
    data = np.fromfile(path, dtype=">i2")
    if data.size != SAMPLES * SAMPLES:
        raise ValueError(f"{path}: {data.size} samples, expected {SAMPLES * SAMPLES} (SRTM3)")
    return data.reshape(SAMPLES, SAMPLES).astype(np.int16)


@dataclass(frozen=True)
class Terrain:
    """A mosaic of tiles. `z[row, col]`, row 0 at latitude `north`, col 0 at longitude `west`."""

    z: np.ndarray  # float32 metres above the EGM96 geoid
    south: float
    west: float
    north: float
    east: float

    @property
    def step(self) -> float:
        """Grid spacing in degrees."""
        return 1.0 / (SAMPLES - 1)

    def contains(self, lat, lon) -> np.ndarray:
        lat = np.asarray(lat, dtype=np.float64)
        lon = np.asarray(lon, dtype=np.float64)
        return (lat >= self.south) & (lat <= self.north) & (lon >= self.west) & (lon <= self.east)

    def elevation(self, lat, lon) -> np.ndarray:
        """Bilinear elevation in metres at the given points. Points outside the mosaic give NaN."""
        lat = np.asarray(lat, dtype=np.float64)
        lon = np.asarray(lon, dtype=np.float64)
        inside = self.contains(lat, lon)
        row = (self.north - lat) / self.step
        col = (lon - self.west) / self.step
        rows, cols = self.z.shape
        r0 = np.clip(np.floor(row).astype(np.int64), 0, rows - 2)
        c0 = np.clip(np.floor(col).astype(np.int64), 0, cols - 2)
        fr = np.clip(row - r0, 0.0, 1.0)
        fc = np.clip(col - c0, 0.0, 1.0)
        top = self.z[r0, c0] * (1 - fc) + self.z[r0, c0 + 1] * fc
        bottom = self.z[r0 + 1, c0] * (1 - fc) + self.z[r0 + 1, c0 + 1] * fc
        value = top * (1 - fr) + bottom * fr
        return np.where(inside, value, np.nan)

    def cell_latlon(self, row, col):
        """Latitude and longitude of grid cells."""
        return self.north - np.asarray(row) * self.step, self.west + np.asarray(col) * self.step


def load_mosaic(folder: str, south: int, west: int, north: int, east: int) -> Terrain:
    """Loads every tile between the integer bounds and joins them into one grid.

    `south` and `west` are the south-west corners of the first tile, `north` and `east`
    the outer edges, so load_mosaic(dir, 25, -101, 28, -99) reads six tiles.
    Voids are filled with the mean of their valid neighbours, repeatedly, so a line
    of sight never crosses a hole of -32768 m.
    """
    n_lat = north - south
    n_lon = east - west
    if n_lat <= 0 or n_lon <= 0:
        raise ValueError("empty bounds")
    size = SAMPLES - 1
    z = np.full((n_lat * size + 1, n_lon * size + 1), VOID, dtype=np.float32)
    for i, lat in enumerate(range(north - 1, south - 1, -1)):
        for j, lon in enumerate(range(west, east)):
            path = os.path.join(folder, tile_name(lat, lon))
            if not os.path.exists(path):
                raise FileNotFoundError(f"missing tile {path}; run scripts/fetch_srtm.py")
            z[i * size : i * size + SAMPLES, j * size : j * size + SAMPLES] = read_tile(path)
    return Terrain(z=_fill_voids(z), south=float(south), west=float(west), north=float(north), east=float(east))


def _fill_voids(z: np.ndarray) -> np.ndarray:
    void = z <= VOID + 1
    if not void.any():
        return z
    z = z.copy()
    z[void] = np.nan
    for _ in range(64):
        holes = np.isnan(z)
        if not holes.any():
            break
        padded = np.pad(z, 1, mode="edge")
        stack = np.stack([padded[:-2, 1:-1], padded[2:, 1:-1], padded[1:-1, :-2], padded[1:-1, 2:]])
        with np.errstate(invalid="ignore"):
            count = np.sum(~np.isnan(stack), axis=0)
            total = np.nansum(stack, axis=0)
        fill = np.where(count > 0, total / np.maximum(count, 1), np.nan)
        z[holes] = fill[holes]
    return np.nan_to_num(z, nan=0.0).astype(np.float32)


def distance_m(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance in metres (haversine)."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


def destination(lat: float, lon: float, bearing_deg, distance) -> tuple[np.ndarray, np.ndarray]:
    """Points at `distance` metres from (lat, lon) along `bearing_deg`.

    A local flat approximation: good to a fraction of a percent over the 150 km this
    project uses, and much faster than the spherical formula.
    """
    bearing = np.radians(bearing_deg)
    dlat = distance * np.cos(bearing) / EARTH_RADIUS_M
    dlon = distance * np.sin(bearing) / (EARTH_RADIUS_M * math.cos(math.radians(lat)))
    return lat + np.degrees(dlat), lon + np.degrees(dlon)
