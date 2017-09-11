#!/usr/bin/env python
"""Test Extent class."""

import os
import sys
import unittest

ROOT_DIR = os.path.split(os.path.abspath(os.path.dirname(__file__)))[0]
sys.path.append(ROOT_DIR)
import heatmap as hm


class Tests(unittest.TestCase):

    def test_no_init(self):
        # Act / Assert
        with self.assertRaises(ValueError):
            hm.Extent()

    def test_init_from_coords(self):
        # Act
        extent = hm.Extent(coords=(hm.Coordinate(-10, -10),
                                   hm.Coordinate(10, 10)))

        # Assert
        self.assertIsNotNone(extent)
        self.assertIsInstance(extent, hm.Extent)
        self.assertEqual(str(extent), "-10,-10,10,10")

        # Assert
        self.assertIsNotNone(extent)
        self.assertIsInstance(extent, hm.Extent)
        self.assertEqual(str(extent), "-10,-10,10,10")

    def test_size(self):
        # Act / Assert
        extent = hm.Extent(coords=(hm.Coordinate(-10, -10),
                                   hm.Coordinate(10, 10)))

        # Assert
        self.assertEqual(extent.size(), hm.Coordinate(20, 20))

    def test_update(self):
        # Arrange
        extent1 = hm.Extent(coords=(hm.Coordinate(-10, -10),
                                    hm.Coordinate(10, 10)))

        extent2 = hm.Extent(coords=(hm.Coordinate(0, 0),
                                    hm.Coordinate(40, 40)))

        # Act
        extent1.update(extent2)

        # Assert
        self.assertEqual(extent1.size(), hm.Coordinate(50, 50))

    def test_from_bounding_box(self):
        # Arrange
        extent1 = hm.Extent(coords=(hm.Coordinate(-10, -10),
                                    hm.Coordinate(10, 10)))

        extent2 = hm.Extent(coords=(hm.Coordinate(0, 0),
                                    hm.Coordinate(40, 40)))

        # Act
        extent1.from_bounding_box(extent2)

        # Assert
        self.assertEqual(extent1.size(), hm.Coordinate(40, 40))

    def test_corners(self):
        # Arrange
        extent = hm.Extent(coords=(hm.Coordinate(-10, -10),
                                   hm.Coordinate(10, 10)))

        # Act / Assert
        self.assertEqual(extent.corners(),
                         (hm.Coordinate(-10, -10),
                          hm.Coordinate(10, 10)))

    def test_grow(self):
        # Arrange
        extent = hm.Extent(coords=(hm.Coordinate(-10, -10),
                                   hm.Coordinate(10, 10)))

        # Act
        extent.grow(20)

        # Assert
        self.assertEqual(extent.corners(),
                         (hm.Coordinate(-30, -30),
                          hm.Coordinate(30, 30)))

    def test_resize(self):
        # Arrange
        extent = hm.Extent(coords=(hm.Coordinate(-10, -10),
                                   hm.Coordinate(10, 10)))

        # Act
        extent.resize(width=10, height=20)

        # Assert
        self.assertEqual(extent.corners(),
                         (hm.Coordinate(-5, -10),
                          hm.Coordinate(5, 10)))

    def test_is_inside(self):
        # Arrange
        extent = hm.Extent(coords=(hm.Coordinate(-10, -10),
                                   hm.Coordinate(10, 10)))

        # Act / Assert
        self.assertTrue(extent.is_inside(hm.Coordinate(1, 1)))
        self.assertFalse(extent.is_inside(hm.Coordinate(100, 100)))


if __name__ == '__main__':
    unittest.main()
