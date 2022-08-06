#!/usr/bin/env python
"""Test random_example.py.  There's nothing worse than broken examples."""

import os
import subprocess
import sys
import unittest
from helper import ROOT_DIR, TEST_DIR


class Tests(unittest.TestCase):
    def test_system(self):
        output_file = os.path.join(TEST_DIR, 'output.ppm')
        try:
            subprocess.check_call(
                [os.path.join(ROOT_DIR, 'random_example.py'),
                 '--output', output_file,
                 '1'])
            self.assertTrue(os.path.exists(output_file))
            self.assertTrue(os.path.isfile(output_file))
            self.assertGreater(os.path.getsize(output_file), 0)
            os.remove(output_file)
        finally:
            try:
                os.remove(output_file)
            except OSError:
                pass  # perhaps it was never created


if __name__ == '__main__':
    unittest.main()
