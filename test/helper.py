#!/usr/bin/env python
"""
Test helpers
"""

import imp
import os
import subprocess
import sys

try:
    import unittest2 as unittest  # Python 2.6
except ImportError:
    import unittest

ROOT_DIR = os.path.split(os.path.abspath(os.path.dirname(__file__)))[0]
sys.path.append(ROOT_DIR)

try:
    imp.find_module("coverage")
    # TODO COVERAGE_ARGS or something
    COVERAGE_CMD = ["coverage", "run", "--append", "--source", "heatmap.py"]
except ImportError:
    COVERAGE_CMD = []


class TestHeatmap(unittest.TestCase):

    def helper_run(self, args):
        # Arrange
        args = list(COVERAGE_CMD) + args

        # Act
        subprocess.check_call(args)

        # Assert
        # Should run with no exceptions

# End of file
