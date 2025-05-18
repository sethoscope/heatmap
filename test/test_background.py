#!/usr/bin/env python
"""Test case."""

import os
import subprocess
import unittest
from helper import TestHeatmap, TEST_DIR


class Tests(TestHeatmap):

    def test_system_background(self):
        '''Test system with a background image.'''
        output_file = os.path.join(TEST_DIR, 'output.png')
        try:
            self.helper_run(
                ['-I', os.path.join(TEST_DIR, 'playa.jpg'),
                 '-r', '10',
                 '-o', output_file,
                 os.path.join(TEST_DIR, 'few-points')])

            subprocess.check_call(
                ['perceptualdiff',
                 os.path.join(TEST_DIR, 'with-background.png'),
                 output_file])
        finally:
            try:
                os.remove(output_file)
            except OSError:
                pass  # perhaps it was never created

    def test_system_background_dark(self):
        '''Test system with a background image.'''
        output_file = os.path.join(TEST_DIR, 'output.png')
        try:
            self.helper_run(
                ['-I', os.path.join(TEST_DIR, 'playa.jpg'),
                 '-r', '10',
                 '-B', '0.4',
                 '-o', output_file,
                 os.path.join(TEST_DIR, 'few-points')])

            subprocess.check_call(
                ['perceptualdiff',
                 os.path.join(TEST_DIR, 'with-background-dark.png'),
                 output_file])
        finally:
            try:
                os.remove(output_file)
            except OSError:
                pass  # perhaps it was never created

    def test_system_background_inverted(self):
        '''Test system with a background image.'''
        output_file = os.path.join(TEST_DIR, 'output.png')
        try:
            self.helper_run(
                ['-I', os.path.join(TEST_DIR, 'playa.jpg'),
                 '-r', '10',
                 '-N',
                 '-o', output_file,
                 os.path.join(TEST_DIR, 'few-points')])

            subprocess.check_call(
                ['perceptualdiff',
                 os.path.join(TEST_DIR, 'with-background-inverted.png'),
                 output_file])
        finally:
            try:
                os.remove(output_file)
            except OSError:
                pass  # perhaps it was never created


if __name__ == '__main__':
    unittest.main()
