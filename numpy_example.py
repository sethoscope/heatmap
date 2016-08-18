#!/usr/bin/env python
#
# Plots some random data to a heatmap.  This is not mainly useful for
# testing and to illustrate how to use heatmap.py from other python code.
#
# Copyright 2014 Seth Golub http://www.sethoscope.net/heatmap/
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General
# Public License along with this program.  If not, see
# <http://www.gnu.org/licenses/>.

from __future__ import print_function

import random
import logging
import sys
import heatmap as hm
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np


def shapes_generator(count):
    for i in range(count):
        a = random.gauss(10, 1.0)
        b = random.gauss(10, 2.0)
        yield hm.Point(hm.LatLon(a, b),)


def setup_config(count):
    config = hm.Configuration()
    config.shapes = shapes_generator(count)
    config.projection = hm.EquirectangularProjection()
    config.projection.pixels_per_degree = 30
    config.decay = 1
    config.kernel = hm.LinearKernel(5)
    config.background = 'black'
    config.fill_missing()
    return config


def matrix_to_numpy(config, matrix):
    extent = config.extent_out or matrix.extent()
    arr = np.zeros((int(extent.size().x) + 1,
                    int(extent.size().y) + 1))
    for (coord, value) in matrix.items():
        x = int(coord.x - extent.min.x)
        y = int(coord.y - extent.min.y)
        if extent.is_inside(coord):
            arr[x, y] = value
            logging.debug('set (%d,%d) to %f' % (x, y, arr[x, y]))
    return arr

def main():
    logging.basicConfig(format='%(relativeCreated)8d ms  // %(message)s')
    description = 'generate random points, save them in a numpy array'
    parser = ArgumentParser(description=description,
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--output', metavar='FILE', default='/tmp/out.png')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('count', type=int)
    args = parser.parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    logging.debug('python version %s' % str(sys.version))

    config = setup_config(args.count)
    matrix = hm.process_shapes(config)
    matrix = matrix.finalized()
    arr = matrix_to_numpy(config, matrix)
    print('shape: ' + str(arr.shape))
    print('max value: %f' % arr.max())
    nonzero = np.count_nonzero(arr)
    print('nonzero cells: %d / %d (%d%%)' % (nonzero, arr.size,
                                             int(100.0 * nonzero / arr.size)))


if __name__ == '__main__':
    main()
