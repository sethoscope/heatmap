#!/usr/bin/env python
"""Test projections."""

import os
import sys

try:
    import unittest2 as unittest  # Python 2.6
except ImportError:
    import unittest

ROOT_DIR = os.path.split(os.path.abspath(os.path.dirname(__file__)))[0]
sys.path.append(ROOT_DIR)
import heatmap as hm


class Tests(unittest.TestCase):
    def _assert_mapping(self, latlon_given, xy_given, projection):
        latlon_given = hm.LatLon(*latlon_given)
        xy_given = hm.Coordinate(*xy_given)

        xy_actual = projection.project(latlon_given)
        self.assertTrue(isinstance(xy_actual, hm.Coordinate))
        self.assertAlmostEqual(xy_given.x, xy_actual.x)
        self.assertAlmostEqual(xy_given.y, xy_actual.y)

        latlon_actual = projection.inverse_project(xy_given)
        self.assertTrue(isinstance(latlon_actual, hm.LatLon))
        self.assertAlmostEqual(latlon_given.lat, latlon_actual.lat)
        self.assertAlmostEqual(latlon_given.lon, latlon_actual.lon)

    def test_equirectangular(self):
        '''Test EquirectangularProjection class.'''
        proj = hm.EquirectangularProjection()
        proj.pixels_per_degree = 1

        self._assert_mapping((0, 0), (0, 0), proj)
        self._assert_mapping((37, -122), (-122, -37), proj)
        self._assert_mapping((-37, 122), (122, 37), proj)

    def test_mercator(self):
        '''Test EquirectangularProjection class.'''
        proj = hm.MercatorProjection()
        proj.pixels_per_degree = 1

        self._assert_mapping((0, 0), (0, 0), proj)
        self._assert_mapping((37, -122), (-122, -39.87717474825904), proj)
        self._assert_mapping((-37, 122), (122, 39.87717474825904), proj)


if __name__ == '__main__':
    unittest.main()
