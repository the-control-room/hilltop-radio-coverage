"""Tests. Run from the repository root: python -m unittest discover -s tests -v"""

import math
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from coverage import corridor, radio  # noqa: E402
from coverage.terrain import SAMPLES, Terrain, distance_m, load_mosaic, tile_name  # noqa: E402

SRTM = os.path.join(ROOT, "data", "srtm")
HAVE_TILES = os.path.exists(os.path.join(SRTM, "N25W101.hgt"))


def flat(height=0.0, bump=None):
    """A one-degree synthetic terrain at the equator-ish latitude 26 to 27, flat at `height`."""
    z = np.full((SAMPLES, SAMPLES), height, dtype=np.float32)
    if bump:
        bump(z)
    return Terrain(z=z, south=26.0, west=-100.0, north=27.0, east=-99.0)


class Formulas(unittest.TestCase):
    def test_radio_horizon_rule_of_thumb(self):
        # The planner's rule: d_km = 4.12 * sqrt(h_m), for k = 4/3.
        for h in (1.5, 15, 100, 1000):
            self.assertAlmostEqual(radio.radio_horizon_m(h) / 1000, 4.12 * math.sqrt(h), delta=0.01 * 4.12 * math.sqrt(h))

    def test_free_space_loss(self):
        # 32.45 + 20 log10(f MHz) + 20 log10(d km): 150 MHz over 1 km is 75.97 dB.
        self.assertAlmostEqual(radio.free_space_loss_db(1000, 150e6), 75.97, places=1)
        # Doubling the distance costs 6.02 dB.
        self.assertAlmostEqual(radio.free_space_loss_db(2000, 150e6) - radio.free_space_loss_db(1000, 150e6), 6.02, places=2)

    def test_fresnel_radius_midpath(self):
        # 150 MHz, 60 km path, midpoint: sqrt(lambda * d / 4) = sqrt(2 * 60000 / 4) = 173 m.
        self.assertAlmostEqual(float(radio.fresnel_radius_m(30_000, 30_000, 150e6)), 173.1, delta=0.5)

    def test_earth_bulge_midpath(self):
        # 50 km path, midpoint, k = 4/3: 25 km * 25 km / (2 * 4/3 * 6371 km) = 36.8 m.
        self.assertAlmostEqual(float(radio.earth_bulge_m(25_000, 25_000)), 36.8, delta=0.1)


class SmoothEarth(unittest.TestCase):
    def test_path_clears_just_inside_the_horizon_and_not_beyond(self):
        terrain = flat()
        site = (26.5, -99.9)
        h1, h2 = 30.0, 1.5
        limit = radio.smooth_earth_range_m(h1, h2)  # about 27.6 km
        metres_per_degree = float(distance_m(26.5, -99.9, 26.5, -98.9))
        inside = -99.9 + 0.97 * limit / metres_per_degree
        outside = -99.9 + 1.03 * limit / metres_per_degree
        result = radio.visible(terrain, site, [26.5, 26.5], [inside, outside], h1, h2, samples=2048)
        self.assertTrue(result[0])
        self.assertFalse(result[1])

    def test_a_ridge_blocks_and_a_taller_mast_clears_it(self):
        def ridge(z):
            z[:, 600:603] = 60.0  # a 60 m wall running north-south through the middle

        terrain = flat(bump=ridge)
        site, target = (26.5, -99.6), (26.5, -99.4)  # about 20 km apart, the wall half way
        self.assertFalse(radio.visible(terrain, site, [target[0]], [target[1]], 15.0, 1.5)[0])
        self.assertTrue(radio.visible(terrain, site, [target[0]], [target[1]], 150.0, 1.5)[0])

    def test_fresnel_test_is_stricter(self):
        terrain = flat()
        site = (26.5, -99.6)
        plain = radio.clearance_m(terrain, site, [26.5], [-99.5], 15.0, 15.0)[0]
        strict = radio.clearance_m(terrain, site, [26.5], [-99.5], 15.0, 15.0, fresnel_hz=150e6)[0]
        self.assertGreater(plain, 0)
        self.assertLess(strict, plain)

    def test_viewshed_on_flat_ground_is_a_disc_of_horizon_radius(self):
        terrain = flat()
        h1, h2 = 30.0, 1.5
        shed = radio.viewshed(terrain, (26.5, -99.5), h1, h2, radius_m=40_000, azimuths=360, step_m=90)
        reach = radio.viewshed_reach_km(shed)
        expected = radio.smooth_earth_range_m(h1, h2) / 1000
        self.assertAlmostEqual(reach, expected, delta=0.3)
        self.assertAlmostEqual(radio.viewshed_area_km2(shed), math.pi * expected**2, delta=0.03 * math.pi * expected**2)


class Cover(unittest.TestCase):
    def test_greedy_takes_the_biggest_gain_first(self):
        cover = np.array([[1, 1, 1, 0, 0, 0], [0, 0, 1, 1, 0, 0], [0, 0, 0, 1, 1, 1]], dtype=bool)
        chosen, history = corridor.greedy_cover(cover)
        self.assertEqual(sorted(chosen), [0, 2])
        self.assertEqual(history[-1], 1.0)

    def test_link_rule_keeps_the_chain_connected(self):
        cover = np.array([[1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0], [0, 0, 0, 0, 1, 1]], dtype=bool)
        # Site 0 sees only site 1, site 1 sees both, site 2 sees only site 1.
        link = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool)
        chosen, _ = corridor.greedy_cover(cover, link)
        self.assertEqual(len(chosen), 3)
        for position in range(1, len(chosen)):
            self.assertTrue(link[chosen[position], chosen[:position]].any())

    def test_densify_spacing_and_length(self):
        route = np.array([[26.0, -100.0], [27.0, -100.0]])
        points, chainage = corridor.densify(route, 1000.0)
        self.assertAlmostEqual(chainage[-1] / 1000, 111.2, delta=0.2)
        self.assertAlmostEqual(float(np.diff(chainage)[0]), 1000.0)


@unittest.skipUnless(HAVE_TILES, "SRTM tiles not downloaded; run scripts/fetch_srtm.py")
class RealTerrain(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.terrain = load_mosaic(SRTM, 25, -101, 28, -99)

    def test_tile_names(self):
        self.assertEqual(tile_name(25, -101), "N25W101.hgt")
        self.assertEqual(tile_name(-3, 17), "S03E017.hgt")

    def test_mosaic_shape_and_known_heights(self):
        self.assertEqual(self.terrain.z.shape, (3 * 1200 + 1, 2 * 1200 + 1))
        # Central Monterrey lies at about 540 m, Nuevo Laredo at about 130 to 150 m.
        self.assertTrue(500 < float(self.terrain.elevation(25.6694, -100.3098)) < 580)
        self.assertTrue(110 < float(self.terrain.elevation(27.4863, -99.5075)) < 160)

    def test_viewshed_agrees_with_point_to_point_check(self):
        site = (26.2, -100.2)
        shed = radio.viewshed(self.terrain, site, 15.0, 1.5, radius_m=60_000, azimuths=720, step_m=90)
        rng = np.random.default_rng(7)
        a = rng.integers(0, 720, 400)
        s = rng.integers(20, shed["distance_m"].size, 400)
        direct = radio.visible(self.terrain, site, shed["lat"][a, s], shed["lon"][a, s], 15.0, 1.5, samples=1024)
        agreement = float(np.mean(direct == shed["visible"][a, s]))
        self.assertGreater(agreement, 0.95)


if __name__ == "__main__":
    unittest.main()
