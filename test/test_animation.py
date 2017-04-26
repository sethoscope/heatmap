#!/usr/bin/env python
"""Animation test case."""

import os
import subprocess
import sys
import unittest

ROOT_DIR = os.path.split(os.path.abspath(os.path.dirname(__file__)))[0]
sys.path.append(ROOT_DIR)


class Tests(unittest.TestCase):

    def test_animation(self):
        '''Currently this only tests whether an output file is created.'''
        output_file = os.path.join(ROOT_DIR, 'test', 'output.mpeg')
        try:
            subprocess.check_call(
                [os.path.join(ROOT_DIR, 'heatmap.py'),
                 '-b', 'black',
                 '-r', '3',
                 '-W', '22',
                 '-P', 'equirectangular',
                 '-a',
                 '--frequency', '1',
                 '-o', output_file,
                 os.path.join(ROOT_DIR, 'test', 'few-points')])

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
