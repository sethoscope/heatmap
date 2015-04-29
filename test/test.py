#!/usr/bin/env python
"""Test case."""

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

    def test_colormap_floats(self):
        self.assertEqual(hm.ColorMap._str_to_float('00'), 0.0)
        self.assertEqual(hm.ColorMap._str_to_float('10'), 0.0625)
        self.assertEqual(hm.ColorMap._str_to_float('100'), 1.0)
        self.assertEqual(hm.ColorMap._str_to_float('110'), 1.0625)
        self.assertEqual(hm.ColorMap._str_to_float('cc'), 0.796875)

    def test_colormap_strings(self):
        self.assertTupleEqual(hm.ColorMap.str_to_hsva('#101010101'),
                              hm.ColorMap.str_to_hsva('101010101'))
        self.assertTupleEqual(hm.ColorMap.str_to_hsva('000000000'),
                              (0.0, 0.0, 0.0, 0.0))
        self.assertTupleEqual(hm.ColorMap.str_to_hsva('000000010'),
                              (0.0, 0.0, 0.0, 0.0625))
        self.assertTupleEqual(hm.ColorMap.str_to_hsva('000001000'),
                              (0.0, 0.0, 0.0625, 0.0))
        self.assertTupleEqual(hm.ColorMap.str_to_hsva('000100000'),
                              (0.0, 0.0625, 0.0, 0.0))
        self.assertTupleEqual(hm.ColorMap.str_to_hsva('010000000'),
                              (0.0625, 0.0, 0.0, 0.0))
        self.assertTupleEqual(hm.ColorMap.str_to_hsva('110000000'),
                              (1.0625, 0.0, 0.0, 0.0))

    def test_system(self):
        output_file = os.path.join(ROOT_DIR, 'test', 'output.ppm')
        save_file = os.path.join(ROOT_DIR, 'test', 'test.pkl')
        try:
            subprocess.check_call(
                [os.path.join(ROOT_DIR, 'heatmap.py'),
                 '-p', os.path.join(ROOT_DIR, 'test', 'few-points'),
                 '-b', 'black',
                 '-r', '3',
                 '-W', '22',
                 '-P', 'equirectangular',
                 '--save', save_file,
                 '-o', output_file])

            subprocess.check_call(
                ['perceptualdiff',
                 os.path.join(ROOT_DIR, 'test', 'few-points.ppm'),
                 output_file])

            os.remove(output_file)
            subprocess.check_call(
                [os.path.join(ROOT_DIR, 'heatmap.py'),
                 '-b', 'black',
                 '-r', '3',
                 '-W', '22',
                 '--load', save_file,
                 '-o', output_file])

            subprocess.check_call(
                ['perceptualdiff',
                 os.path.join(ROOT_DIR, 'test', 'few-points.ppm'),
                 output_file])

        finally:
            try:
                os.remove(output_file)
            except OSError:
                pass  # perhaps it was never created

            try:
                os.remove(save_file)
            except OSError:
                pass  # perhaps it was never created

    def test__scale_for_osm_zoom(self):
        # Arrange
        zoom = 8

        # Act
        scale = hm._scale_for_osm_zoom(zoom)

        # Assert
        self.assertEqual(scale, 182.04444444444445)

    def test_choose_osm_zoom_with_no_zoom_w_h_in_config(self):
        # Arrange
        config = hm.Configuration(use_defaults=True)
        padding = 2

        # Act
        # Act / Assert
        with self.assertRaisesRegex(ValueError,
                                    "For OSM, you must specify height, "
                                    "width, or zoom"):
            hm.choose_osm_zoom(config, padding)

    def test_choose_osm_zoom_with_zoom_in_config(self):
        # Arrange
        config = hm.Configuration(use_defaults=True)
        config.zoom = 2
        padding = 2

        # Act
        zoom = hm.choose_osm_zoom(config, padding)

        # Assert
        self.assertEqual(zoom, 2)

    def test_choose_osm_zoom_with_w_h_in_config(self):
        # Arrange
        config = hm.Configuration(use_defaults=True)
        config.width = 400
        config.height = 200
        config.extent_in = hm.Extent(coords=(hm.LatLon(-10, -10),
                                             hm.LatLon(10, 10)))
        padding = 2

        # Act
        zoom = hm.choose_osm_zoom(config, padding)

        # Assert
        self.assertEqual(zoom, 3)

if __name__ == '__main__':
    unittest.main()
