# How much road can one hilltop see?

Line-of-sight radio coverage over real terrain, written for a video by The Control Room about a
private two-way radio network in Mexico.

The video: https://www.youtube.com/watch?v=d_ldxFRyxxk

A court record describes that network as hand-held radios linked by towers and repeaters. It gives
no tower locations. So this code does not reconstruct the network. It answers a narrower question:
**what does the terrain allow?** Every site in the output is picked by the code from elevation data
alone. None is a known or alleged antenna location.

## What it finds

Mexican Federal Highway 85 from Monterrey to the Texas border, 245 km in the road data used here.
Assumptions: repeater antennas 15 m above the ground, hand-held radios 1.5 m above the ground,
standard 4/3 Earth radius.

| Question | Result |
| --- | --- |
| Two hand-held radios on flat ground lose sight of each other at | 10.1 km |
| People needed to pass a message down the road with radios only | 25 |
| The best single hilltop within 15 km of the road stands | 952 m above the road, 9.0 km from it |
| Road in sight of one mast on that hilltop | 176.8 km, 72% |
| The same mast standing beside the road instead | 15.9 km |
| Road in sight with 2, 7, 10 and 20 linked sites | 77%, 92%, 96%, 99% |

The first hilltop does most of the work. The flat ground near the border has no hills to stand on,
so the last few per cent need more masts than the first seventy.

## Run it

```bash
pip install -r requirements.txt
python scripts/fetch_srtm.py
python -m unittest discover -s tests
python run.py
```

`run.py` prints the numbers and writes `out/results.json`, a shaded-relief picture and one
visibility mask per site. It takes about a minute.

## How it works

- `coverage/terrain.py` reads SRTM `.hgt` tiles, joins six of them and samples heights.
- `coverage/radio.py` holds the geometry. A path is clear when the straight line between two antennas
  stays above the ground after the ground has been raised by the Earth's bulge, `d1 * d2 / (2 k R)`
  with `k = 4/3`. `viewshed()` sweeps rays outwards and keeps the steepest angle to the ground so
  far. An optional stricter test also wants 60 per cent of the first Fresnel zone clear.
- `coverage/corridor.py` looks for candidate sites (the highest point of every 2 km block within
  15 km of the road, ranked by how far it stands above its surroundings, plus a mast beside the road
  every 5 km) and picks sites greedily: each time the one that adds the most road, as long as it can
  see a site already chosen, so the result is one linked chain.
- `tests/` checks the formulas against textbook values, the line-of-sight test against a smooth
  Earth and a synthetic ridge, and the ray sweep against the point-to-point test on real terrain.

## What this is not

- Not a map of any real network. The court record behind the video gives no locations.
- Not a coverage prediction. Line of sight is the first-order limit for VHF and UHF, but real
  coverage also depends on power, antennas, vegetation, buildings and diffraction over ridges. A
  5 W hand-held radio has roughly 40 to 60 dB of margin on a clear 10 to 100 km path at 150 MHz, so
  on these paths the horizon is the limit, not power. That is as far as the claim goes.
- Not an optimum. Greedy set cover gives a good chain, not a proven minimum.
- The road geometry is small-scale (Natural Earth 1:10m) and follows the real road to within a few
  kilometres.

## Data

- Elevation: SRTM3 version 2.1, three arc-seconds, from NASA's Shuttle Radar Topography Mission,
  distributed by the USGS. Public domain. The tiles are not in this repository; `fetch_srtm.py`
  downloads them from the Internet Archive's copy of the retired USGS server and checks their
  SHA-256 sums.
- Road: `data/route_mx85.geojson`, extracted from Natural Earth 1:10m roads (public domain) by
  `scripts/extract_route.py`.

## Licence

MIT, see `LICENSE`. The data keeps its own terms as described above.
