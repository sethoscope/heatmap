#!/usr/bin/env python
"""Test case."""

import os
import subprocess

from helper import TestHeatmap, unittest, ROOT_DIR


class Tests(TestHeatmap):

    def test_negative_values(self):
        output_file = os.path.join(ROOT_DIR, 'test', 'output.ppm')
        try:
            self.helper_run(
                [os.path.join(ROOT_DIR, 'heatmap.py'),
                 '-b', 'black',
                 '-r', '3',
                 '-W', '22',
                 '-o', output_file,
                 os.path.join(ROOT_DIR, 'test', 'negative-values.txt')])

        finally:
            try:
                os.remove(output_file)
            except OSError:
                pass  # perhaps it was never created

    def test_negative_and_positive_values(self):
        output_file = os.path.join(ROOT_DIR, 'test', 'output.ppm')
        try:
            self.helper_run(
                [os.path.join(ROOT_DIR, 'heatmap.py'),
                 '-b', 'black',
                 '-r', '3',
                 '-W', '22',
                 '-o', output_file,
                 os.path.join(ROOT_DIR, 'test', 'negative-and-positive-values.txt')])

        finally:
            try:
                os.remove(output_file)
            except OSError:
                pass  # perhaps it was never created

    def test_system(self):
        output_file = os.path.join(ROOT_DIR, 'test', 'output.ppm')
        save_file = os.path.join(ROOT_DIR, 'test', 'test.pkl')
        try:
            self.helper_run(
                [os.path.join(ROOT_DIR, 'heatmap.py'),
                 '-b', 'black',
                 '-r', '3',
                 '-W', '22',
                 '-P', 'equirectangular',
                 '--save', save_file,
                 '-o', output_file,
                 os.path.join(ROOT_DIR, 'test', 'few-points')])

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


if __name__ == '__main__':
    unittest.main()
