#!/usr/bin/env python
"""Test coordinate classes."""

import os
import sys
import unittest
import helper
import heatmap as hm


class Tests(unittest.TestCase):

    # To remove Python 3's
    # "DeprecationWarning: Please use assertRaisesRegex instead"
    if sys.version_info[0] == 2:
        assertRaisesRegex = unittest.TestCase.assertRaisesRegexp

    def test_basic(self):
        '''Test Configuration class.'''
        # Act
        config = hm.Configuration(use_defaults=True)

        # Assert
        self.assertEqual(config.margin, 0)
        self.assertEqual(config.frequency, 1)

    def test_fill_missing_no_input(self):
        '''Test Configuration class.'''
        # Arrange
        config = hm.Configuration(use_defaults=True)

        # Act / Assert
        with self.assertRaisesRegex(ValueError, "no input specified"):
            config.fill_missing()


if __name__ == '__main__':
    unittest.main()
