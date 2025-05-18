#!/usr/bin/env python
"""Test case."""

import os
import subprocess
import unittest
from helper import TestHeatmap, TEST_DIR


class Tests(TestHeatmap):

    def test_system_gradient(self):
        '''Test system with a gradient image file.'''
        output_file = os.path.join(TEST_DIR, 'output.ppm')
        try:
            self.helper_run(
                ['-b', 'black',
                 '-r', '3',
                 '-W', '22',
                 '-P', 'equirectangular',
                 '-G', os.path.join(TEST_DIR, 'gradient.png'),
                 '-o', output_file,
                 os.path.join(TEST_DIR, 'few-points')])

            subprocess.check_call(
                ['perceptualdiff',
                 os.path.join(TEST_DIR, 'few-points.ppm'),
                 output_file])
        finally:
            try:
                os.remove(output_file)
            except OSError:
                pass  # perhaps it was never created


if __name__ == '__main__':
    unittest.main()
