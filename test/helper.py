#!/usr/bin/env python
"""
Test helpers
"""

import os
import subprocess
import sys
import unittest
import heatmap77 as hm

ROOT_DIR = os.path.split(os.path.abspath(os.path.dirname(__file__)))[0]
TEST_DIR = os.path.join(ROOT_DIR, 'test')
EXAMPLES_DIR = os.path.join(ROOT_DIR, 'examples')
HEATMAP_PY = os.path.join(hm.__path__[0], 'heatmap.py')

try:
    import coverage
    _ = coverage.__version__  # so pyflakes doesn't warn about unused import
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
