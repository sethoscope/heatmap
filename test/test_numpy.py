#!/usr/bin/env python
"""Test numpy_example.py.  There's nothing worse than broken examples."""

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
        # We just check that it runs, not that it's correct.
        subprocess.check_call(
            [os.path.join(ROOT_DIR, 'numpy_example.py'),
             '100'])

if __name__ == '__main__':
    unittest.main()
