#!/usr/bin/env python
"""Test coordinate classes."""

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

    def test_basic(self):
        '''Test Coordinate class.'''
        coord = hm.Coordinate(1, 2)
        self.assertEqual(coord.x, 1)
        self.assertEqual(coord.y, 2)
        self.assertEqual(coord.first, coord.x)
        self.assertEqual(coord.second, coord.y)
        self.assertEqual(coord, coord.copy())
        self.assertEqual(str(coord), "(1, 2)")
        self.assertIsInstance(hash(coord), int)
        coord2 = hm.Coordinate(6, 12)
        self.assertEqual(coord2 - coord, hm.Coordinate(5, 10))

    def test_latlon(self):
        '''Test LatLon class.'''
        coord = hm.LatLon(37, -122)
        self.assertEqual(coord.lat, 37)
        self.assertEqual(coord.lon, -122)
        self.assertEqual(coord.y, 37)
        self.assertEqual(coord.x, -122)

        self.assertEqual(coord, coord.copy())

        # test order
        self.assertEqual(coord.first, coord.lat)
        self.assertEqual(coord.second, coord.lon)

        # test updates
        coord.lat = 25
        self.assertEqual(coord.lat, 25)
        self.assertEqual(coord.y, 25)
        coord.y = 10
        self.assertEqual(coord.lat, 10)
        self.assertEqual(coord.y, 10)


if __name__ == '__main__':
    unittest.main()
