#!/usr/bin/env python
"""Test case."""

from helper import TestHeatmap, unittest, ROOT_DIR, TEST_DIR

import os
import subprocess


class Tests(TestHeatmap):
    def test_points(self):
        output_file = os.path.join(TEST_DIR, 'output.ppm')
        try:
            self.helper_run(
                ['-p', os.path.join(TEST_DIR, 'few-points'),
                 '-b', 'black',
                 '-r', '3',
                 '-W', '22',
                 '-P', 'equirectangular',
                 '-o', output_file])

            subprocess.check_call(
                ['perceptualdiff',
                 os.path.join(TEST_DIR, 'few-points.ppm'),
                 output_file])
        finally:
            try:
                os.remove(output_file)
            except OSError:
                pass  # perhaps it was never created

    def test_csv(self):
        output_file = os.path.join(TEST_DIR, 'output.ppm')
        try:
            self.helper_run(
                ['--csv', os.path.join(TEST_DIR, 'few-points.csv'),
                 '--ignore_csv_header',
                 '-b', 'black',
                 '-r', '3',
                 '-W', '22',
                 '-P', 'equirectangular',
                 '-o', output_file])

            subprocess.check_call(
                ['perceptualdiff',
                 os.path.join(TEST_DIR, 'few-points.ppm'),
                 output_file])
        finally:
            try:
                os.remove(output_file)
            except OSError:
                pass  # perhaps it was never created

    def test_gpx(self):
        output_file = os.path.join(TEST_DIR, 'output.ppm')
        try:
            self.helper_run(
                ['-g', os.path.join(ROOT_DIR, 'test', 'smile.gpx'),
                 '-b', 'black',
                 '-r', '3',
                 '-W', '22',
                 '-P', 'equirectangular',
                 '-o', output_file])

            subprocess.check_call(
                ['perceptualdiff',
                 os.path.join(TEST_DIR, 'smile-gpx.ppm'),
                 output_file])
        finally:
            try:
                os.remove(output_file)
            except OSError:
                pass  # perhaps it was never created

    def test_shp(self):
        output_file = os.path.join(TEST_DIR, 'output.ppm')
        try:
            self.helper_run(['--shp_file', os.path.join(TEST_DIR,
                                                        'test_shape_2.shp'),
                             '-b', 'black',
                             '-r', '3',
                             '-W', '22',
                             '-o', output_file])

        finally:
            try:
                os.remove(output_file)
            except OSError:
                pass  # perhaps it was never created


if __name__ == '__main__':
    unittest.main()
