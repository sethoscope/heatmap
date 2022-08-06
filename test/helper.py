#!/usr/bin/env python
"""
Test helpers
"""

import os
import subprocess
import sys
import unittest

ROOT_DIR = os.path.split(os.path.abspath(os.path.dirname(__file__)))[0]
TEST_DIR = os.path.join(ROOT_DIR, 'test')
sys.path.append(ROOT_DIR)
HEATMAP_PY = os.path.join(ROOT_DIR, 'heatmap.py')

try:
    import coverage
    COVERAGE_CMD = [sys.executable,
                    "-m", "coverage", "run",
                    "--append", "--include=heatmap.py",
                    HEATMAP_PY]
except ImportError:
    COVERAGE_CMD = [sys.executable, HEATMAP_PY]


class TestHeatmap(unittest.TestCase):
    def helper_run(self, args):
        subprocess.check_call(COVERAGE_CMD + args)


# End of file
