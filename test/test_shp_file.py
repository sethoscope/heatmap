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


class Tests(unittest.TestCase):

    def test_system(self):
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
