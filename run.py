"""Runs the three experiments and writes out/results.json plus the map layers.

Usage, from the repository root:  python run.py

1. Smooth Earth: how far radios reach when nothing is in the way but the planet.
2. One repeater: the best single hilltop near Highway 85, on real terrain.
3. The corridor: how many linked repeaters it takes to cover the road from Monterrey
   to the border.

Every site in the output is chosen by this script from elevation data alone. None is a
known or alleged antenna location. The question answered is what the terrain allows.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np
from PIL import Image

from coverage import corridor, radio
from coverage.terrain import distance_m, load_mosaic

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "out")

# Assumptions. Change them and run again.
MAST_M = 15.0  # repeater antenna height above the hilltop
HANDHELD_M = 1.5  # a radio held by a standing person
VEHICLE_MAST_M = 15.0  # the same mast standing on the road instead of a hill
FREQUENCY_HZ = 150e6  # VHF high band, typical for commercial two-way radio
HANDHELD_W = 5.0  # typical handheld transmit power
SENSITIVITY_DBM = -116.0  # typical 12 dB SINAD sensitivity of an FM receiver
ROUTE_SPACING_M = 500.0
ROADSIDE_SPACING_M = 5_000.0  # a mast beside the road is a candidate site every 5 km
SITE_SEARCH_M = 15_000.0  # hilltops this close to the road are candidates
BOUNDS = (25, -101, 28, -99)  # south, west, north, east


def image_grid(terrain, pixels_per_degree_lat=560):
    """A lat/lon grid for pictures: equirectangular, squeezed by cos(latitude) so shapes look right."""
    mid = math.radians((terrain.south + terrain.north) / 2)
    height = int(round((terrain.north - terrain.south) * pixels_per_degree_lat))
    width = int(round((terrain.east - terrain.west) * pixels_per_degree_lat * math.cos(mid)))
    lat = terrain.north - (np.arange(height) + 0.5) * (terrain.north - terrain.south) / height
    lon = terrain.west + (np.arange(width) + 0.5) * (terrain.east - terrain.west) / width
    return np.meshgrid(lat, lon, indexing="ij"), width, height


def hillshade(z, cell_m=90.0, azimuth=315.0, altitude=45.0, exaggeration=2.0):
    gy, gx = np.gradient(z * exaggeration, cell_m)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    az, alt = math.radians(azimuth), math.radians(altitude)
    shade = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    return np.clip(shade, 0, 1)


def write_relief(terrain, grid, name, low, high):
    """Shaded relief in two given colours (low ground to high ground), as an RGB PNG."""
    (lat, lon), _w, _h = grid
    z = terrain.elevation(lat, lon)
    cell = float(distance_m(lat[0, 0], lon[0, 0], lat[1, 0], lon[0, 0]))
    shade = hillshade(z, cell)
    level = np.clip((z - 100.0) / 2400.0, 0, 1) ** 0.6
    low = np.array(low, dtype=np.float64)
    high = np.array(high, dtype=np.float64)
    base = low[None, None, :] + (high - low)[None, None, :] * level[:, :, None]
    lit = base * (0.62 + 0.5 * shade[:, :, None])
    Image.fromarray(np.clip(lit, 0, 255).astype(np.uint8)).save(os.path.join(OUT, name))


def raster_mask(grid, site, shed):
    """The polar viewshed, resampled onto the picture grid."""
    (lat, lon), _w, _h = grid
    lat0, lon0 = site
    north = (lat - lat0) * math.pi / 180 * 6_371_000.0
    east = (lon - lon0) * math.pi / 180 * 6_371_000.0 * math.cos(math.radians(lat0))
    dist = np.hypot(north, east)
    bearing = (np.degrees(np.arctan2(east, north)) + 360.0) % 360.0
    step = shed["distance_m"][1] - shed["distance_m"][0]
    s = np.round(dist / step).astype(np.int64) - 1
    a = np.round(bearing / (360.0 / len(shed["bearing_deg"]))).astype(np.int64) % len(shed["bearing_deg"])
    ok = (s >= 0) & (s < shed["distance_m"].size)
    mask = np.zeros(lat.shape, dtype=bool)
    mask[ok] = shed["visible"][a[ok], s[ok]]
    return mask


def write_mask(mask, name, rgb=(31, 63, 255), alpha=150):
    rgba = np.zeros(mask.shape + (4,), dtype=np.uint8)
    rgba[mask] = (*rgb, alpha)
    Image.fromarray(rgba, "RGBA").save(os.path.join(OUT, name))


def runs_of(covered, chainage):
    """Longest stretch of road without coverage, in km."""
    longest = current = 0.0
    for index in range(1, len(covered)):
        if not covered[index]:
            current += chainage[index] - chainage[index - 1]
            longest = max(longest, current)
        else:
            current = 0.0
    return longest / 1000.0


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    terrain = load_mosaic(os.path.join(ROOT, "data", "srtm"), *BOUNDS)
    route = corridor.load_route(os.path.join(ROOT, "data", "route_mx85.geojson"))
    points, chainage = corridor.densify(route, ROUTE_SPACING_M)
    route_km = chainage[-1] / 1000.0
    results: dict = {
        "assumptions": {
            "mast_m": MAST_M,
            "handheld_m": HANDHELD_M,
            "k_factor": radio.K_FACTOR,
            "frequency_mhz": FREQUENCY_HZ / 1e6,
            "handheld_w": HANDHELD_W,
            "sensitivity_dbm": SENSITIVITY_DBM,
            "route_spacing_m": ROUTE_SPACING_M,
            "site_search_km": SITE_SEARCH_M / 1000.0,
            "terrain": "SRTM3 v2.1 (NASA/USGS), 3 arc-seconds",
            "route": "Natural Earth 1:10m roads, Mexican Federal Highway 85/85D",
        },
        "route_km": round(route_km, 1),
    }

    # 1. Smooth Earth.
    smooth = {
        "handheld_to_handheld_km": radio.smooth_earth_range_m(HANDHELD_M, HANDHELD_M) / 1000,
        "handheld_to_mast_km": radio.smooth_earth_range_m(HANDHELD_M, MAST_M) / 1000,
        "handheld_to_mast_on_500m_hill_km": radio.smooth_earth_range_m(HANDHELD_M, 500 + MAST_M) / 1000,
        "handheld_to_mast_on_1000m_hill_km": radio.smooth_earth_range_m(HANDHELD_M, 1000 + MAST_M) / 1000,
    }
    results["smooth_earth"] = {key: round(value, 1) for key, value in smooth.items()}
    results["smooth_earth"]["handheld_hops_for_route"] = math.ceil(route_km / smooth["handheld_to_handheld_km"])

    # Is power the limit, or the horizon? Margin of a 5 W handheld into a repeater on a clear path.
    results["link_margin_db"] = {
        f"{km}_km": round(radio.link_margin_db(km * 1000, FREQUENCY_HZ, HANDHELD_W, 0.0, 6.0, SENSITIVITY_DBM), 1) for km in (10, 50, 100)
    }

    # 2. Candidate hilltops and what each sees of the road.
    hilltops = corridor.candidate_hilltops(terrain, points, within_m=SITE_SEARCH_M)
    # Flat country has no hilltops, so a mast beside the road is a candidate too: one every 5 km.
    roadside_index = np.arange(0, len(points), int(ROADSIDE_SPACING_M / ROUTE_SPACING_M))
    roadside = np.column_stack([points[roadside_index, 0], points[roadside_index, 1], terrain.elevation(points[roadside_index, 0], points[roadside_index, 1])])
    sites = np.vstack([hilltops, roadside])
    kind = ["hilltop"] * len(hilltops) + ["roadside"] * len(roadside)
    cover = corridor.coverage_matrix(terrain, sites, points, MAST_M, HANDHELD_M)
    best = int(cover[: len(hilltops)].sum(axis=1).argmax())
    lat0, lon0, elev0 = sites[best]
    shed = radio.viewshed(terrain, (lat0, lon0), MAST_M, HANDHELD_M)
    nearest_road = float(distance_m(lat0, lon0, points[:, 0], points[:, 1]).min())
    road_at_site = points[int(distance_m(lat0, lon0, points[:, 0], points[:, 1]).argmin())]
    results["single_site"] = {
        "lat": round(float(lat0), 4),
        "lon": round(float(lon0), 4),
        "ground_m": round(float(elev0)),
        "height_above_road_m": round(float(elev0 - terrain.elevation(road_at_site[0], road_at_site[1]))),
        "distance_to_road_km": round(nearest_road / 1000, 1),
        "road_km_in_sight": round(float(cover[best].mean() * route_km), 1),
        "road_share_in_sight": round(float(cover[best].mean()), 3),
        "area_km2": round(radio.viewshed_area_km2(shed)),
        "area_note": "counted inside the six tiles and within 150 km only",
        "farthest_road_point_km": round(float(distance_m(lat0, lon0, points[cover[best], 0], points[cover[best], 1]).max()) / 1000, 1),
        "p95_ground_km": round(radio.viewshed_reach_km(shed, 95), 1),
        "hilltops_examined": int(len(hilltops)),
    }

    # The same mast standing on the road itself, at the road point nearest the best hilltop.
    valley = radio.visible(terrain, (road_at_site[0], road_at_site[1]), points[:, 0], points[:, 1], VEHICLE_MAST_M, HANDHELD_M)
    results["mast_on_the_road"] = {"road_km_in_sight": round(float(valley.mean() * route_km), 1)}

    # 3. The corridor.
    link = corridor.link_matrix(terrain, sites, MAST_M)
    chosen, history = corridor.greedy_cover(cover, link, target=1.0)
    chosen_free, history_free = corridor.greedy_cover(cover, None, target=1.0)
    covered = cover[chosen].any(axis=0)
    chain = []
    seen = np.zeros(len(points), dtype=bool)
    for order, index in enumerate(chosen):
        seen |= cover[index]
        chain.append(
            {
                "order": order + 1,
                "lat": round(float(sites[index, 0]), 4),
                "lon": round(float(sites[index, 1]), 4),
                "ground_m": round(float(sites[index, 2])),
                "kind": kind[index],
                "road_km_added_total": round(float(seen.mean() * route_km), 1),
                "share_total": round(float(seen.mean()), 3),
                "links_to": [int(chosen.index(other) + 1) for other in chosen[:order] if link[index, other]],
            }
        )
    hops = [float(distance_m(sites[a, 0], sites[a, 1], sites[b, 0], sites[b, 1])) / 1000 for a in chosen for b in chosen if a < b and link[a, b]]
    results["corridor"] = {
        "sites": chain,
        "count_linked": len(chosen),
        "hilltops_in_chain": sum(1 for index in chosen if kind[index] == "hilltop"),
        "roadside_masts_in_chain": sum(1 for index in chosen if kind[index] == "roadside"),
        "share_linked": round(history[-1], 3) if history else 0.0,
        "count_unlinked": len(chosen_free),
        "share_unlinked": round(history_free[-1], 3) if history_free else 0.0,
        "share_after_each_site": [round(share, 3) for share in history],
        "sites_for_90_percent": next((i + 1 for i, share in enumerate(history) if share >= 0.9), None),
        "sites_for_95_percent": next((i + 1 for i, share in enumerate(history) if share >= 0.95), None),
        "longest_gap_km": round(runs_of(covered, chainage), 1),
        "longest_link_km": round(max(hops), 1) if hops else None,
    }

    # A terrain profile for the picture: from the best hilltop to the farthest road point it sees.
    visible_points = np.flatnonzero(cover[best])
    far = visible_points[int(distance_m(lat0, lon0, points[visible_points, 0], points[visible_points, 1]).argmax())]
    dist, ground = radio.profile(terrain, lat0, lon0, points[far, 0], points[far, 1], samples=400)
    bulge = radio.earth_bulge_m(dist, dist[-1] - dist)
    line = (ground[0] + MAST_M) + ((ground[-1] + HANDHELD_M) - (ground[0] + MAST_M)) * (dist / dist[-1])
    results["profile"] = {
        "km": [round(float(v) / 1000, 2) for v in dist],
        "ground_plus_bulge_m": [round(float(v), 1) for v in ground + bulge],
        "line_m": [round(float(v), 1) for v in line],
        "length_km": round(float(dist[-1]) / 1000, 1),
        "min_clearance_m": round(float(np.min((line - ground - bulge)[1:-1])), 1),
    }
    # And a blocked one: the nearest road point the hilltop cannot see beyond its first 5 km.
    hidden = np.flatnonzero(~cover[best] & (distance_m(lat0, lon0, points[:, 0], points[:, 1]) > 5000))
    if hidden.size:
        near = hidden[int(distance_m(lat0, lon0, points[hidden, 0], points[hidden, 1]).argmin())]
        d2, g2 = radio.profile(terrain, lat0, lon0, points[near, 0], points[near, 1], samples=400)
        b2 = radio.earth_bulge_m(d2, d2[-1] - d2)
        l2 = (g2[0] + MAST_M) + ((g2[-1] + HANDHELD_M) - (g2[0] + MAST_M)) * (d2 / d2[-1])
        results["profile_blocked"] = {
            "km": [round(float(v) / 1000, 2) for v in d2],
            "ground_plus_bulge_m": [round(float(v), 1) for v in g2 + b2],
            "line_m": [round(float(v), 1) for v in l2],
            "length_km": round(float(d2[-1]) / 1000, 1),
        }

    # Pictures.
    grid = image_grid(terrain)
    (_lat, _lon), width, height = grid
    write_relief(terrain, grid, "relief-light.png", low=(244, 246, 248), high=(100, 116, 139))
    write_relief(terrain, grid, "relief-dark.png", low=(30, 41, 59), high=(148, 163, 184))
    write_mask(raster_mask(grid, (lat0, lon0), shed), "viewshed-single.png")
    union = np.zeros((height, width), dtype=bool)
    for order, index in enumerate(chosen):
        s = radio.viewshed(terrain, (sites[index, 0], sites[index, 1]), MAST_M, HANDHELD_M)
        mask = raster_mask(grid, (sites[index, 0], sites[index, 1]), s)
        write_mask(mask & ~union, f"viewshed-chain-{order + 1}.png")
        union |= mask

    def to_pixel(lat, lon):
        return [round((lon - terrain.west) / (terrain.east - terrain.west) * width, 1), round((terrain.north - lat) / (terrain.north - terrain.south) * height, 1)]

    results["picture"] = {
        "width": width,
        "height": height,
        "south": terrain.south,
        "west": terrain.west,
        "north": terrain.north,
        "east": terrain.east,
        "km_per_pixel": round(111.19 * (terrain.north - terrain.south) / height, 4),
        "route_px": [to_pixel(lat, lon) for lat, lon in points[::4]],
        "route_covered_single": [bool(v) for v in cover[best][::4]],
        "route_covered_chain": [bool(v) for v in covered[::4]],
        "single_site_px": to_pixel(lat0, lon0),
        "chain_px": [to_pixel(site["lat"], site["lon"]) for site in chain],
        "chain_kind": [site["kind"] for site in chain],
        "hilltops_px": [to_pixel(lat, lon) for lat, lon, _elev in hilltops],
        "places_px": {"Monterrey": to_pixel(25.6866, -100.3161), "Nuevo Laredo": to_pixel(27.4763, -99.5164), "Sabinas Hidalgo": to_pixel(26.5086, -100.1789)},
    }

    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=1)
    summary = {key: value for key, value in results.items() if key not in ("profile", "profile_blocked", "picture")}
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
