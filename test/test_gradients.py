#!/usr/bin/env python
"""Test case."""

import os
import subprocess
import sys
import unittest

ROOT_DIR = os.path.split(os.path.abspath(os.path.dirname(__file__)))[0]
sys.path.append(ROOT_DIR)


class Tests(unittest.TestCase):

    def test_system_gradient(self):
        '''Test system with a gradient image file.'''
        output_file = os.path.join(ROOT_DIR, 'test', 'output.ppm')
        try:
            subprocess.check_call(
                [os.path.join(ROOT_DIR, 'heatmap.py'),
                 '-b', 'black',
                 '-r', '3',
                 '-W', '22',
                 '-P', 'equirectangular',
                 '-G', os.path.join(ROOT_DIR, 'test', 'gradient.png'),
                 '-o', output_file,
                 os.path.join(ROOT_DIR, 'test', 'few-points')])

            subprocess.check_call(
                ['perceptualdiff',
                 os.path.join(ROOT_DIR, 'test', 'few-points.ppm'),
                 output_file])
        finally:
            try:
                os.remove(output_file)
            except OSError:
                pass  # perhaps it was never created


if __name__ == '__main__':
    unittest.main()
