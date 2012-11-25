#!/usr/bin/env python
"""Test case."""

import os
import subprocess
import sys


ROOT_DIR = os.path.split(os.path.abspath(os.path.dirname(__file__)))[0]


def main():
    subprocess.check_call([os.path.join(ROOT_DIR, 'heatmap.py'),
                           '-b', 'black',
                           '-p', os.path.join(ROOT_DIR, 'test', 'graffiti.coords'),
                           '-r', '30', '-W', '300',
                           '-o', os.path.join(ROOT_DIR, 'test', 'g1.ppm'),
                           '-P', 'equirectangular'])

    subprocess.check_call(['perceptualdiff',
                           os.path.join(ROOT_DIR, 'test', 'graffiti.ppm'),
                           os.path.join(ROOT_DIR, 'test', 'g1.ppm')])


if __name__ == '__main__':
    sys.exit(main())
