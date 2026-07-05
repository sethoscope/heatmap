#!/usr/bin/env python
"""Test case."""

import os
import subprocess
import unittest
from helper import TestHeatmap, TEST_DIR


class Tests(TestHeatmap):

    def test_negative_values(self):
        output_file = os.path.join(TEST_DIR, 'output.ppm')
        try:
            self.helper_run(
                ['-b', 'black',
                 '-r', '3',
                 '-W', '22',
                 '-o', output_file,
                 os.path.join(TEST_DIR, 'negative-values.csv')])

        finally:
            try:
                os.remove(output_file)
            except OSError:
                pass  # perhaps it was never created

    def test_negative_and_positive_values(self):
        output_file = os.path.join(TEST_DIR, 'output.ppm')
        try:
            self.helper_run(
                ['-b', 'black',
                 '-r', '3',
                 '-W', '22',
                 '-o', output_file,
                 os.path.join(TEST_DIR, 'negative-and-positive-values.csv')])

        finally:
            try:
                os.remove(output_file)
            except OSError:
                pass  # perhaps it was never created

    def test_system(self):
        output_file = os.path.join(TEST_DIR, 'output.ppm')
        try:
            self.helper_run(
                ['-b', 'black',
                 '-r', '3',
                 '-W', '22',
                 '-P', 'equirectangular',
                 '-o', output_file,
                 '--ignore_csv_header',
                 os.path.join(TEST_DIR, 'few-points.csv')])

            subprocess.check_call(
                ['perceptualdiff',
                 os.path.join(TEST_DIR, 'few-points.ppm'),
                 output_file])
        finally:
            try:
                os.remove(output_file)
            except OSError:
                pass  # perhaps it was never created

    def test_weights(self):
        output_file = os.path.join(ROOT_DIR, 'test', 'output.png')
        try:
            self.helper_run(
                [os.path.join(ROOT_DIR, 'heatmap.py'),
                 '-W', '640',
                 '-d', '0.20',
                 '-o', output_file,
                 '--ignore_csv_header',
                 os.path.join(ROOT_DIR, 'test', 'weights.csv')])

            subprocess.check_call(
                ['perceptualdiff',
                 os.path.join(ROOT_DIR, 'test', 'weights.png'),
                 output_file])
        finally:
            try:
                os.remove(output_file)
            except OSError:
                pass  # perhaps it was never created

if __name__ == '__main__':
    unittest.main()
