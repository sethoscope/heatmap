#!/usr/bin/env python
"""Test projection scale."""

import os
import subprocess
import sys

try:
    import unittest2 as unittest  # Python 2.6
except ImportError:
    import unittest

ROOT_DIR = os.path.split(os.path.abspath(os.path.dirname(__file__)))[0]
sys.path.append(ROOT_DIR)
import heatmap as hm


class Tests(unittest.TestCase):

    # To remove Python 3's
    # "DeprecationWarning: Please use assertRaisesRegex instead"
    if sys.version_info[0] == 2:
        assertRaisesRegex = unittest.TestCase.assertRaisesRegexp

    def test_units(self):
        '''Test projection units.'''
        p = hm.Projection()
        p.pixels_per_degree = 2
        self.assertAlmostEqual(p.pixels_per_degree, 2.0)
        self.assertAlmostEqual(p.meters_per_pixel, 55659.7453966)

        p.meters_per_pixel = 500
        self.assertAlmostEqual(p.pixels_per_degree, 222.63898158654)
        self.assertAlmostEqual(p.meters_per_pixel, 500)

    def test_system(self):
        output_file_1 = os.path.join(ROOT_DIR, 'test', 'output-1.ppm')
        output_file_2 = os.path.join(ROOT_DIR, 'test', 'output-2.ppm')
        try:
            subprocess.check_call(
                [os.path.join(ROOT_DIR, 'heatmap.py'),
                 '-p', os.path.join(ROOT_DIR, 'test', 'few-points'),
                 '-b', 'black',
                 '-r', '3',
                 '-P', 'equirectangular',
                 '--scale', '55659.745397',
                 '-o', output_file_1])

            subprocess.check_call(
                [os.path.join(ROOT_DIR, 'heatmap.py'),
                 '-p', os.path.join(ROOT_DIR, 'test', 'few-points'),
                 '-b', 'black',
                 '-r', '3',
                 '-P', 'equirectangular',
                 '-W', '22',
                 '-H', '16',
                 '-o', output_file_2])

            subprocess.check_call(
                ['perceptualdiff',
                 output_file_1,
                 output_file_2])

        finally:
            try:
                os.remove(output_file_1)
                os.remove(output_file_2)
            except OSError:
                pass  # perhaps it was never created

    def test_has_pixels_per_degree(self):
        # Arrange
        p = hm.Projection()
        p.pixels_per_degree = 2

        # Act
        scaled = p.is_scaled()
        ppd = p.get_pixels_per_degree()

        # Assert
        self.assertTrue(scaled)
        self.assertEqual(ppd, 2)

    def test_no_pixels_per_degree(self):
        # Arrange
        p = hm.Projection()

        # Act
        scaled = p.is_scaled()

        # Assert
        self.assertFalse(scaled)
        with self.assertRaisesRegex(AttributeError,
                                    "projection scale was never set"):
            p.get_pixels_per_degree()

    def test_not_implemented(self):
        # Arrange
        p = hm.Projection()
        coords = 0

        # Act / Assert
        with self.assertRaisesRegex(NotImplementedError, ""):
            p.project(coords)
        with self.assertRaisesRegex(NotImplementedError, ""):
            p.inverse_project(coords)


if __name__ == '__main__':
    unittest.main()
