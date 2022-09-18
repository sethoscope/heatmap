#!/usr/bin/env python3
#
# Generates two heat maps from the same data and composites them together.
#
# Copyright 2022 Seth Golub http://www.sethoscope.net/heatmap/
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

import logging
import sys
import heatmap as hm
import PIL
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

# This is just an example of doing two heatmaps and compositing the
# results, all in one python process.  Some of the config is here in
# code, and some needs to be passed on the command line, which is not
# ideal. It could all be moved to new command line arguments with
# defaults, or could all get put into code.
#
# If run with these arguments:
#
#  -W 1200 -o out.png -P equirectangular test-data/graffiti.coords
#
# it produces output similar to the graffiti example on the website.


def main():
    logging.basicConfig(format='%(relativeCreated)8d ms  // %(message)s')
    config = hm.Configuration(use_defaults=False)
    parser = config.argparser
    parser.description = 'make a 2-pass composite heatmap'

    args = parser.parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    config.set_from_options(args)
    config.fill_missing()

    logging.debug('python version %s' % str(sys.version))

    config.decay = 0.3
    config.kernel = hm.LinearKernel(5)
    config.background = None
    matrix = hm.process_shapes(config)
    matrix = matrix.finalized()
    image1 = hm.ImageMaker(config).make_image(matrix)

    config.decay = 0.95
    config.kernel = hm.LinearKernel(30)
    config.background = 'black'
    matrix = hm.process_shapes(config)
    matrix = matrix.finalized()
    image2 = hm.ImageMaker(config).make_image(matrix)
    image2.putalpha(255)
    logging.debug(f'image2.mode: {image2.mode}')

    image = PIL.Image.alpha_composite(image2, image1)
    image.save(args.output)


if __name__ == '__main__':
    main()
