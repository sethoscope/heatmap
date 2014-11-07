#!/usr/bin/env python
"""Test random_example.py.  There's nothing worse than broken examples."""

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
