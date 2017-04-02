#!/usr/bin/env python
"""Test case."""

from helper import TestHeatmap, unittest, ROOT_DIR

import os
import subprocess


class Tests(TestHeatmap):
    def test_points(self):
        output_file = os.path.join(ROOT_DIR, 'test', 'output.ppm')
        try:
            self.helper_run(
                [os.path.join(ROOT_DIR, 'heatmap.py'),
                 '-p', os.path.join(ROOT_DIR, 'test', 'few-points'),
                 '-b', 'black',
                 '-r', '3',
                 '-W', '22',
                 '-P', 'equirectangular',
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

    def test_csv(self):
        output_file = os.path.join(ROOT_DIR, 'test', 'output.ppm')
        try:
            self.helper_run(
                [os.path.join(ROOT_DIR, 'heatmap.py'),
                 '--csv', os.path.join(ROOT_DIR, 'test', 'few-points.csv'),
                 '--ignore_csv_header',
                 '-b', 'black',
                 '-r', '3',
                 '-W', '22',
                 '-P', 'equirectangular',
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

    def test_gpx(self):
        output_file = os.path.join(ROOT_DIR, 'test', 'output.ppm')
        try:
            self.helper_run(
                [os.path.join(ROOT_DIR, 'heatmap.py'),
                 '-g', os.path.join(ROOT_DIR, 'test', 'smile.gpx'),
                 '-b', 'black',
                 '-r', '3',
                 '-W', '22',
                 '-P', 'equirectangular',
                 '-o', output_file])

            subprocess.check_call(
                ['perceptualdiff',
                 os.path.join(ROOT_DIR, 'test', 'smile-gpx.ppm'),
                 output_file])
        finally:
            try:
                os.remove(output_file)
            except OSError:
                pass  # perhaps it was never created

    def test_shp(self):
        output_file = os.path.join(ROOT_DIR, 'test', 'output.ppm')
        try:
            subprocess.check_call(
                [os.path.join(ROOT_DIR, 'heatmap.py'),
                 '--shp_file', os.path.join(ROOT_DIR, 'test',
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
