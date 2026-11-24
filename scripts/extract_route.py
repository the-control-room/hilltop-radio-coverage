"""Builds data/route_mx85.geojson from Natural Earth's 1:10m roads (public domain).

Usage: python scripts/extract_route.py <path to ne_10m_roads.geojson>

The file is at
https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_10m_roads.geojson
(about 50 MB). The route is Mexican Federal Highway 85 and its toll twin 85D between
Monterrey and Nuevo Laredo. Natural Earth is a small-scale dataset: the line follows the
road to within a few kilometres, which is enough for an example calculation.
"""

import json
import math
import os
import sys

BOX = (-101.0, 25.5, -99.0, 27.8)  # west, south, east, north
START = (-100.32, 25.67)  # Monterrey end of the chain


def main(source: str) -> None:
    with open(source, encoding="utf-8") as handle:
        features = json.load(handle)["features"]

    lines = []
    for feature in features:
        props = feature["properties"]
        geometry = feature["geometry"]
        if not geometry or props.get("sov_a3") != "MEX" or props.get("name") not in ("85", "85D"):
            continue
        parts = [geometry["coordinates"]] if geometry["type"] == "LineString" else geometry["coordinates"]
        for part in parts:
            if all(BOX[0] <= x <= BOX[2] and BOX[1] <= y <= BOX[3] for x, y in part):
                lines.append([(float(x), float(y)) for x, y in part])

    # Chain the pieces end to end, starting from the piece nearest Monterrey.
    def gap(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    chain = []
    here = START
    while lines:
        best = min(range(len(lines)), key=lambda i: min(gap(here, lines[i][0]), gap(here, lines[i][-1])))
        piece = lines.pop(best)
        if gap(here, piece[-1]) < gap(here, piece[0]):
            piece.reverse()
        if gap(here, piece[0]) > 0.08 and chain:
            continue  # a parallel alternative of a stretch already taken
        chain.extend(piece if not chain else piece[1:])
        here = chain[-1]

    out = {
        "type": "Feature",
        "properties": {
            "name": "Mexican Federal Highway 85 / 85D, Monterrey to Nuevo Laredo",
            "source": "Natural Earth 1:10m roads, public domain (naturalearthdata.com)",
            "note": "Small-scale geometry, accurate to a few kilometres.",
        },
        "geometry": {"type": "LineString", "coordinates": [[round(x, 5), round(y, 5)] for x, y in chain]},
    }
    target = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "route_mx85.geojson")
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=1)
    print(f"{len(chain)} vertices, from {chain[0]} to {chain[-1]} -> {target}")


if __name__ == "__main__":
    main(sys.argv[1])
