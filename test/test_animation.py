#!/usr/bin/env python
"""Animation test case."""

import os
import subprocess
import sys
import unittest
from helper import TestHeatmap, TEST_DIR


class Tests(TestHeatmap):

    def test_animation(self):
        '''Currently this only tests whether an output file is created.'''
        output_file = os.path.join(TEST_DIR, 'output.mpeg')
        try:
            self.helper_run(
                ['-b', 'black',
                 '-r', '3',
                 '-W', '22',
                 '-P', 'equirectangular',
                 '-a',
                 '--frequency', '1',
                 '-o', output_file,
                 os.path.join(TEST_DIR, 'few-points')])

            self.assertTrue(os.path.exists(output_file))
            self.assertTrue(os.path.isfile(output_file))
            self.assertGreater(os.path.getsize(output_file), 0)

        finally:
            try:
                os.remove(output_file)
            except OSError:
                pass  # perhaps it was never created


if __name__ == '__main__':
    unittest.main()
