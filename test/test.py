#!/usr/bin/env python
"""Test case."""

import os
import subprocess
import sys


ROOT_DIR = os.path.split(os.path.abspath(os.path.dirname(__file__)))[0]

def main():

    subprocess.check_call([os.path.join(ROOT_DIR, 'heatmap.py'),
                           '-p', os.path.join(ROOT_DIR, 'test', 'few-points'),
                           '-b', 'black',
                           '-r', '3',
                           '-W', '22',
                           '-P', 'equirectangular',
                           '-o', os.path.join(ROOT_DIR, 'test', 'output.ppm')])

    subprocess.check_call(['perceptualdiff',
                           os.path.join(ROOT_DIR, 'test', 'few-points.ppm'),
                           os.path.join(ROOT_DIR, 'test', 'output.ppm')])


if __name__ == '__main__':
    sys.exit(main())
