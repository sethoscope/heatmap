#!/usr/bin/env python
#
# heatmap.py - Generates heat map images and animations from geographic data
# Copyright 2010 Seth Golub
# http://www.sethoscope.net/heatmap/
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import print_function

import sys
import logging
import math
from PIL import Image
from PIL import ImageColor
import tempfile
import os.path
import shutil
import subprocess
from collections import defaultdict
import xml.etree.cElementTree as ET
from colorsys import hsv_to_rgb
try:
    import cPickle as pickle
except ImportError:
    import pickle


class Coordinate(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    first = property(lambda self: self.x)
    second = property(lambda self: self.y)

    def copy(self):
        return self.__class__(self.first, self.second)

    def __str__(self):
        return '(%s, %s)' % (str(self.x), str(self.y))

    def __hash__(self):
        return hash((self.x, self.y))

    def __eq__(self, o):
        return True if self.x == o.x and self.y == o.y else False

    def __sub__(self, o):
        return self.__class__(self.first - o.first, self.second - o.second)


class LatLon(Coordinate):
    def __init__(self, lat, lon):
        self.lat = lat
        self.lon = lon

    def get_lat(self):
        return self.y

    def set_lat(self, lat):
        self.y = lat

    def get_lon(self):
        return self.x

    def set_lon(self, lon):
        self.x = lon

    lat = property(get_lat, set_lat)
    lon = property(get_lon, set_lon)

    first = property(get_lat)
    second = property(get_lon)


class TrackLog:
    class Trkseg(list):  # for GPX <trkseg> tags
        pass

    class Trkpt:  # for GPX <trkpt> tags
        def __init__(self, lat, lon):
            self.coords = LatLon(float(lat), float(lon))

        def __str__(self):
            return str(self.coords)

    def _parse(self, filename):
        self._segments = []
        for event, elem in ET.iterparse(filename, ('start', 'end')):
            elem.tag = elem.tag[elem.tag.rfind('}') + 1:]   # remove namespace
            if elem.tag == "trkseg":
                if event == 'start':
                    self._segments.append(TrackLog.Trkseg())
                else:  # event == 'end'
                    yield self._segments.pop()
                    elem.clear()  # delete contents from parse tree
            elif elem.tag == 'trkpt' and event == 'end':
                point = TrackLog.Trkpt(elem.attrib['lat'], elem.attrib['lon'])
                self._segments[-1].append(point)
                elem.clear()  # clear the trkpt node to minimize memory usage

    def __init__(self, filename):
        self.filename = filename

    def segments(self):
        '''Parse file and yield segments containing points'''
        logging.info('reading GPX track from %s' % self.filename)
        return self._parse(self.filename)


class Projection(object):
    # For guessing scale, we pretend the earth is a sphere with this
    # radius in meters, as in Web Mercator (the projection all the
    # online maps use).
    EARTH_RADIUS = 6378137  # in meters

    def get_pixels_per_degree(self):
        try:
            return self._pixels_per_degree
        except AttributeError:
            raise AttributeError('projection scale was never set')

    def set_pixels_per_degree(self, val):
        self._pixels_per_degree = val
        logging.info('scale: %f meters/pixel (%f pixels/degree)'
                     % (self.meters_per_pixel, val))

    def get_meters_per_pixel(self):
        return 2 * math.pi * self.EARTH_RADIUS / 360 / self.pixels_per_degree

    def set_meters_per_pixel(self, val):
        self.pixels_per_degree = 2 * math.pi * self.EARTH_RADIUS / 360 / val
        return val

    pixels_per_degree = property(get_pixels_per_degree, set_pixels_per_degree)
    meters_per_pixel = property(get_meters_per_pixel, set_meters_per_pixel)

    def is_scaled(self):
        return hasattr(self, '_pixels_per_degree')

    def project(self, coords):
        raise NotImplementedError

    def inverse_project(self, coords):   # Not all projections can do this.
        raise NotImplementedError

    def auto_set_scale(self, extent_in, padding, width=None, height=None):
        # We need to choose a scale at which the data's bounding box,
        # once projected onto the map, will fit in the specified height
        # and/or width.  The catch is that we can't project until we
        # have a scale, so what we'll do is set a provisional scale,
        # project the bounding box onto the map, then adjust the scale
        # appropriately.  This way we don't need to know anything about
        # the projection.
        #
        # Projection subclasses are free to override this method with
        # something simpler that just solves for scale given the lat/lon
        # and x/y bounds.

        # We'll work large to minimize roundoff error.
        SCALE_FACTOR = 1000000.0
        self.pixels_per_degree = SCALE_FACTOR
        extent_out = extent_in.map(self.project)
        padding *= 2  # padding-per-edge -> padding-in-each-dimension
        try:
            if height:
                self.pixels_per_degree = pixels_per_lat = (
                    float(height - padding) /
                    extent_out.size().y * SCALE_FACTOR)
            if width:
                self.pixels_per_degree = (
                    float(width - padding) /
                    extent_out.size().x * SCALE_FACTOR)
                if height:
                    self.pixels_per_degree = min(self.pixels_per_degree,
                                                 pixels_per_lat)
        except ZeroDivisionError:
            raise ZeroDivisionError(
                'You need at least two data points for auto scaling. '
                'Try specifying the scale explicitly (or extent + '
                'height or width).')
        assert(self.pixels_per_degree > 0)


# Treats Lat/Lon as a square grid.
class EquirectangularProjection(Projection):
    # http://en.wikipedia.org/wiki/Equirectangular_projection
    def project(self, coord):
        x = coord.lon * self.pixels_per_degree
        y = -coord.lat * self.pixels_per_degree
        return Coordinate(x, y)

    def inverse_project(self, coord):
        lat = -coord.y / self.pixels_per_degree
        lon = coord.x / self.pixels_per_degree
        return LatLon(lat, lon)


class MercatorProjection(Projection):
    def set_pixels_per_degree(self, val):
        super(MercatorProjection, self).set_pixels_per_degree(val)
        self._pixels_per_radian = val * (180 / math.pi)
    pixels_per_degree = property(Projection.get_pixels_per_degree,
                                 set_pixels_per_degree)

    def project(self, coord):
        x = coord.lon * self.pixels_per_degree
        y = -self._pixels_per_radian * math.log(
            math.tan((math.pi/4 + math.pi/360 * coord.lat)))
        return Coordinate(x, y)

    def inverse_project(self, coord):
        lat = (360 / math.pi *
               math.atan(math.exp(-coord.y / self._pixels_per_radian)) - 90)
        lon = coord.x / self.pixels_per_degree
        return LatLon(lat, lon)


class Extent():
    def __init__(self, coords=None, shapes=None):
        if coords:
            coords = tuple(coords)  # if it's a generator, slurp them all
            self.min = coords[0].__class__(min(c.first for c in coords),
                                           min(c.second for c in coords))
            self.max = coords[0].__class__(max(c.first for c in coords),
                                           max(c.second for c in coords))
        elif shapes:
            self.from_shapes(shapes)
        else:
            raise ValueError('Extent must be initialized')

    def __str__(self):
        return '%s,%s,%s,%s' % (self.min.y, self.min.x, self.max.y, self.max.x)

    def update(self, other):
        '''grow this bounding box so that it includes the other'''
        self.min.x = min(self.min.x, other.min.x)
        self.min.y = min(self.min.y, other.min.y)
        self.max.x = max(self.max.x, other.max.x)
        self.max.y = max(self.max.y, other.max.y)

    def from_bounding_box(self, other):
        self.min = other.min.copy()
        self.max = other.max.copy()

    def from_shapes(self, shapes):
        shapes = iter(shapes)
        self.from_bounding_box(next(shapes).extent)
        for s in shapes:
            self.update(s.extent)

    def corners(self):
        return (self.min, self.max)

    def size(self):
        return self.max.__class__(self.max.x - self.min.x,
                                  self.max.y - self.min.y)

    def grow(self, pad):
        self.min.x -= pad
        self.min.y -= pad
        self.max.x += pad
        self.max.y += pad

    def resize(self, width=None, height=None):
        if width:
            self.max.x += float(width - self.size().x) / 2
            self.min.x = self.max.x - width
        if height:
            self.max.y += float(height - self.size().y) / 2
            self.min.y = self.max.y - height

    def is_inside(self, coord):
        return (coord.x >= self.min.x and coord.x <= self.max.x and
                coord.y >= self.min.y and coord.y <= self.max.y)

    def map(self, func):
        '''Returns a new Extent whose corners are a function of the
        corners of this one.  The expected use is to project a Extent
        onto a map.  For example: bbox_xy = bbox_ll.map(projector.project)'''
        return Extent(coords=(func(self.min), func(self.max)))


class Matrix(defaultdict):
    '''An abstract sparse matrix, with data stored as {coord : value}.'''

    @staticmethod
    def matrix_factory(decay):
        # If decay is 0 or 1, we can accumulate as we go and save lots of
        # memory.
        if decay == 1.0:
            logging.info('creating a summing matrix')
            return SummingMatrix()
        elif decay == 0.0:
            logging.info('creating a maxing matrix')
            return MaxingMatrix()
        logging.info('creating an appending matrix')
        return AppendingMatrix(decay)

    def __init__(self, default_factory=float):
        self.default_factory = default_factory

    def add(self, coord, val):
        raise NotImplementedError

    def extent(self):
        return(Extent(coords=self.keys()))

    def finalized(self):
        return self


class SummingMatrix(Matrix):
    def add(self, coord, val):
        self[coord] += val


class MaxingMatrix(Matrix):
    def add(self, coord, val):
        self[coord] = max(val, self.get(coord, val))


class AppendingMatrix(Matrix):
    def __init__(self, decay):
        self.default_factory = list
        self.decay = decay

    def add(self, coord, val):
        self[coord].append(val)

    def finalized(self):
        logging.info('combining coincident points')
        m = Matrix()
        for (coord, values) in self.items():
            m[coord] = self.reduce(self.decay, values)
        return m

    @staticmethod
    def reduce(decay, values):
        '''
        Returns a weighted sum of the values, where weight N is
        pow(decay,N).  This means the largest value counts fully, but
        additional values have diminishing contributions. decay=0 makes
        the reduction equivalent to max(), which makes each data point
        visible, but says nothing about their relative magnitude.
        decay=1 makes this like sum(), which makes the relative
        magnitude of the points more visible, but could make smaller
        values hard to see.  Experiment with values between 0 and 1.
        Values outside that range will give weird results.
        '''
        # It would be nice to do this on the fly, while accumulating data, but
        # it needs to be insensitive to data order.
        weight = 1.0
        total = 0.0
        values.sort(reverse=True)
        for value in values:
            total += value * weight
            weight *= decay
        return total


class Point:
    def __init__(self, coord, weight=1.0):
        self.coord = coord
        self.weight = weight

    def __str__(self):
        return 'P(%s)' % str(self.coord)

    @staticmethod
    def general_distance(x, y):
        # assumes square units, which causes distortion in some projections
        return (x ** 2 + y ** 2) ** 0.5

    @property
    def extent(self):
        if not hasattr(self, '_extent'):
            self._extent = Extent(coords=(self.coord,))
        return self._extent

    # From a modularity standpoint, it would be reasonable to cache
    # distances, not heat values, and let the kernel cache the
    # distance to heat map, but this is substantially faster.
    heat_cache = {}

    @classmethod
    def _initialize_heat_cache(cls, kernel):
        cache = {}
        for x in range(kernel.radius + 1):
            for y in range(kernel.radius + 1):
                cache[(x, y)] = kernel.heat(cls.general_distance(x, y))
        cls.heat_cache[kernel] = cache

    def add_heat_to_matrix(self, matrix, kernel):
        if kernel not in Point.heat_cache:
            Point._initialize_heat_cache(kernel)
        cache = Point.heat_cache[kernel]
        x = int(self.coord.x)
        y = int(self.coord.y)
        for dx in range(-kernel.radius, kernel.radius + 1):
            for dy in range(-kernel.radius, kernel.radius + 1):
                matrix.add(Coordinate(x + dx, y + dy),
                           self.weight * cache[(abs(dx), abs(dy))])

    def map(self, func):
        return Point(func(self.coord), self.weight)


class LineSegment:
    def __init__(self, start, end, weight=1.0):
        self.start = start
        self.end = end
        self.weight = weight
        self.length_squared = float((self.end.x - self.start.x) ** 2 +
                                    (self.end.y - self.start.y) ** 2)
        self.extent = Extent(coords=(start, end))

    def __str__(self):
        return 'LineSegment(%s, %s)' % (self.start, self.end)

    def distance(self, coord):
        # http://stackoverflow.com/questions/849211/shortest-distance-between-a-point-and-a-line-segment
        # http://www.topcoder.com/tc?d1=tutorials&d2=geometry1&module=Static#line_point_distance
        # http://local.wasp.uwa.edu.au/~pbourke/geometry/pointline/
        try:
            dx = (self.end.x - self.start.x)
            dy = (self.end.y - self.start.y)
            u = ((coord.x - self.start.x) * dx +
                 (coord.y - self.start.y) * dy) / self.length_squared
            if u < 0:
                u = 0
            elif u > 1:
                u = 1
        except ZeroDivisionError:
            u = 0  # Our line is zero-length.  That's ok.
        dx = self.start.x + u * dx - coord.x
        dy = self.start.y + u * dy - coord.y
        return math.sqrt(dx * dx + dy * dy)

    def add_heat_to_matrix(self, matrix, kernel):
        # Iterate over every point in a bounding box around this, with an
        # extra margin given by the kernel's self-reported maximum range.
        # TODO: There is probably a more clever iteration that skips more
        # of the empty space.
        for x in range(int(self.extent.min.x - kernel.radius),
                       int(self.extent.max.x + kernel.radius + 1)):
            for y in range(int(self.extent.min.y - kernel.radius),
                           int(self.extent.max.y + kernel.radius + 1)):
                coord = Coordinate(x, y)
                heat = kernel.heat(self.distance(coord))
                if heat:
                    matrix.add(coord, self.weight * heat)

    def map(self, func):
        return LineSegment(func(self.start), func(self.end))


class LinearKernel:
    '''Uses a linear falloff, essentially turning a point into a cone.'''
    def __init__(self, radius):
        self.radius = radius  # in pixels
        self.radius_float = float(radius)  # worthwhile time saver

    def heat(self, distance):
        if distance >= self.radius:
            return 0.0
        return 1.0 - (distance / self.radius_float)


class GaussianKernel:
    def __init__(self, radius):
        '''radius is the distance beyond which you should not bother.'''
        self.radius = radius
        # We set the scale such that the heat value drops to 1/256 of
        # the peak at a distance of radius.
        self.scale = math.log(256) / radius

    def heat(self, distance):
        '''Returns 1.0 at center, 1/e at radius pixels from center.'''
        if distance >= self.radius:
            return 0.0
        return math.e ** (-distance * self.scale)


class ColorMap:
    DEFAULT_HSVA_MIN_STR = '000ffff00'
    DEFAULT_HSVA_MAX_STR = '02affffff'

    @staticmethod
    def _str_to_float(string, base=16, maxval=256):
        return float(int(string, base)) / maxval

    @staticmethod
    def str_to_hsva(string):
        '''
        Returns a 4-tuple of ints from a hex string color specification,
        such that AAABBCCDD becomes AAA, BB, CC, DD.  For example,
        str2hsva('06688bbff') returns (102, 136, 187, 255).  Note that
        the first number is 3 digits.
        '''
        if string.startswith('#'):
            string = string[1:]  # Leading "#" is now optional.
        return tuple(ColorMap._str_to_float(s) for s in (string[0:3],
                                                         string[3:5],
                                                         string[5:7],
                                                         string[7:9]))

    def __init__(self, hsva_min=None, hsva_max=None, image=None, steps=256):
        '''
        Create a color map based on a progression in the specified
        range, or using pixels in a provided image.

        If supplied, hsva_min and hsva_max must each be a 4-tuple of
        (hue, saturation, value, alpha), where each is a float from
        0.0 to 1.0.  The gradient will be a linear progression from
        hsva_min to hsva_max, including both ends of the range.

        The optional steps argument specifies how many discrete steps
        there should be in the color gradient when using hsva_min
        and hsva_max.
        '''
        # TODO: do the interpolation in Lab space instead of HSV
        self.values = []
        if image:
            assert image.mode == 'RGBA', (
                'Gradient image must be RGBA.  Yours is %s.' % image.mode)
            num_rows = image.size[1]
            self.values = [image.getpixel((0, row)) for row in range(num_rows)]
            self.values.reverse()
            if self.values[0][3] != 0:
                logging.warn('In gradient image %s, the bottom-left pixel is '
                             'not fully transparent. If the output appears '
                             'blocky, make sure your gradient image '
                             'transitions to fully transparent at the bottom.'
                             % os.path.basename(image.filename))
            if self.values[-1][3] != 255:
                logging.warn('In gradient image %s, the top-left pixel is '
                             'not fully opaque. If the output appears '
                             'dim, try increasing the opacity of the '
                             'upper region of your gradient image.'
                             % os.path.basename(image.filename))
        else:
            if not hsva_min:
                hsva_min = ColorMap.str_to_hsva(self.DEFAULT_HSVA_MIN_STR)
            if not hsva_max:
                hsva_max = ColorMap.str_to_hsva(self.DEFAULT_HSVA_MAX_STR)
            # Turn (h1,s1,v1,a1), (h2,s2,v2,a2) into (h2-h1,s2-s1,v2-v1,a2-a1)
            hsva_range = list(map(lambda min, max: max - min,
                                  hsva_min, hsva_max))
            for value in range(0, steps):
                hsva = list(map(
                    lambda range, min: value / float(steps - 1) * range + min,
                    hsva_range, hsva_min))
                hsva[0] = hsva[0] % 1  # in case hue is out of range
                rgba = tuple(
                    [int(x * 255)
                     for x in hsv_to_rgb(*hsva[0:3]) + (hsva[3],)])
                self.values.append(rgba)

    def get(self, floatval):
        return self.values[int(floatval * (len(self.values) - 1))]


class ImageMaker():
    def __init__(self, config):
        '''Each argument to the constructor should be a 4-tuple of (hue,
        saturaton, value, alpha), one to use for minimum data values and
        one for maximum.  Each should be in [0,1], however because hue is
        circular, you may specify hue in any range and it will be shifted
        into [0,1] as needed.  This is so you can wrap around the color
        wheel in either direction.'''
        self.config = config
        if config.background and not config.background_image:
            self.background = ImageColor.getrgb(config.background)
        else:
            self.background = None

    @staticmethod
    def _blend_pixels(a, b):
        # a is RGBA, b is RGB; we could write this more generically,
        # but why complicate things?
        alpha = a[3] / 255.0
        return tuple(
            map(lambda aa, bb: int(aa * alpha + bb * (1 - alpha)), a[:3], b))

    def make_image(self, matrix):
        extent = self.config.extent_out
        if not extent:
            extent = matrix.extent()
        extent.resize((self.config.width or 1) - 1,
                      (self.config.height or 1) - 1)
        size = extent.size()
        size.x = int(size.x) + 1
        size.y = int(size.y) + 1
        logging.info('saving image (%d x %d)' % (size.x, size.y))
        if self.background:
            img = Image.new('RGB', (size.x, size.y), self.background)
        else:
            img = Image.new('RGBA', (size.x, size.y))

        maxval = max(matrix.values())
        logging.info('maximum density: %f' % maxval)
        pixels = img.load()
        for (coord, val) in matrix.items():
            if val == 0.0:
                continue
            x = int(coord.x - extent.min.x)
            y = int(coord.y - extent.min.y)
            if extent.is_inside(coord):
                color = self.config.colormap.get(val / maxval)
                if self.background:
                    pixels[x, y] = ImageMaker._blend_pixels(color,
                                                            self.background)
                else:
                    pixels[x, y] = color
        if self.config.background_image:
            img = Image.composite(img, self.config.background_image,
                                  img.split()[3])
        return img


class ImageSeriesMaker():
    '''Creates a movie showing the data appearing on the heatmap.'''
    def __init__(self, config):
        self.config = config
        self.image_maker = ImageMaker(config)
        self.tmpdir = tempfile.mkdtemp()
        self.imgfile_template = os.path.join(self.tmpdir, 'frame-%05d.png')

    def _save_image(self, matrix):
        self.frame_count += 1
        logging.info('Frame %d' % (self.frame_count))
        matrix = matrix.finalized()
        image = self.image_maker.make_image(matrix)
        image.save(self.imgfile_template % self.frame_count)

    def maybe_save_image(self, matrix):
        self.inputs_since_output += 1
        if self.inputs_since_output >= self.config.frequency:
            self._save_image(matrix)
            self.inputs_since_output = 0

    @staticmethod
    def create_movie(infiles, outfile, ffmpegopts):
        command = ['ffmpeg', '-i', infiles]
        if ffmpegopts:
            # I hope they don't have spaces in their arguments
            command.extend(ffmpegopts.split())
        command.append(outfile)
        logging.info('Encoding video: %s' % ' '.join(command))
        subprocess.call(command)

    def run(self):
        logging.info('Putting animation frames in %s' % self.tmpdir)
        self.inputs_since_output = 0
        self.frame_count = 0
        matrix = process_shapes(self.config, self.maybe_save_image)
        if ((not self.frame_count or
             self.inputs_since_output >= self.config.straggler_threshold)):
            self._save_image(matrix)
        self.create_movie(self.imgfile_template,
                          self.config.output,
                          self.config.ffmpegopts)
        if self.config.keepframes:
            logging.info('The animation frames are in %s' % self.tmpdir)
        else:
            shutil.rmtree(self.tmpdir)
        return matrix


def _get_osm_image(bbox, zoom, osm_base):
    # Just a wrapper for osm.createOSMImage to translate coordinate schemes
    try:
        from osmviz.manager import PILImageManager, OSMManager
        osm = OSMManager(
            image_manager=PILImageManager('RGB'),
            server=osm_base)
        (c1, c2) = bbox.corners()
        image, bounds = osm.createOSMImage((c1.lat, c2.lat, c1.lon, c2.lon),
                                           zoom)
        (lat1, lat2, lon1, lon2) = bounds
        return image, Extent(coords=(LatLon(lat1, lon1),
                                     LatLon(lat2, lon2)))
    except ImportError as e:
        logging.error(
            "ImportError: %s.\n"
            "The --osm option depends on the osmviz module, available from\n"
            "http://cbick.github.com/osmviz/\n\n" % str(e))
        sys.exit(1)


def _scale_for_osm_zoom(zoom):
    return 256 * pow(2, zoom) / 360.0


def choose_osm_zoom(config, padding):
    # Since we know we're only going to do this with Mercator, we could do
    # a bit more math and solve this directly, but as a first pass method,
    # we instead project the bounding box into pixel-land at a high zoom
    # level, then see the power of two we're off by.
    if config.zoom:
        return config.zoom
    if not (config.width or config.height):
        raise ValueError('For OSM, you must specify height, width, or zoom')
    crazy_zoom_level = 30
    proj = MercatorProjection()
    scale = _scale_for_osm_zoom(crazy_zoom_level)
    proj.pixels_per_degree = scale
    bbox_crazy_xy = config.extent_in.map(proj.project)
    if config.width:
        size_ratio = width_ratio = (
            float(bbox_crazy_xy.size().x) / (config.width - 2 * padding))
    if config.height:
        size_ratio = (
            float(bbox_crazy_xy.size().y) / (config.height - 2 * padding))
        if config.width:
            size_ratio = max(size_ratio, width_ratio)
    # TODO: We use --height and --width as upper bounds, choosing a zoom
    # level that lets our image be no larger than the specified size.
    # It might be desirable to use them as lower bounds or to get as close
    # as possible, whether larger or smaller (where "close" probably means
    # in pixels, not scale factors).
    # TODO: This is off by a little bit at small scales.
    zoom = int(crazy_zoom_level - math.log(size_ratio, 2))
    logging.info('Choosing OSM zoom level %d' % zoom)
    return zoom


def get_osm_background(config, padding):
    zoom = choose_osm_zoom(config, padding)
    proj = MercatorProjection()
    proj.pixels_per_degree = _scale_for_osm_zoom(zoom)
    bbox_xy = config.extent_in.map(proj.project)
    # We're not checking that the padding fits within the specified size.
    bbox_xy.grow(padding)
    bbox_ll = bbox_xy.map(proj.inverse_project)
    image, img_bbox_ll = _get_osm_image(bbox_ll, zoom, config.osm_base)
    img_bbox_xy = img_bbox_ll.map(proj.project)

    # TODO: this crops to our data extent, which means we're not making
    # an image of the requested dimensions.  Perhaps we should let the
    # user specify whether to treat the requested size as min,max,exact.
    offset = bbox_xy.min - img_bbox_xy.min
    image = image.crop((
        int(offset.x),
        int(offset.y),
        int(offset.x + bbox_xy.size().x + 1),
        int(offset.y + bbox_xy.size().y + 1)))
    config.background_image = image
    config.extent_in = bbox_ll
    config.projection = proj
    (config.width, config.height) = image.size
    return image, bbox_ll, proj


def process_shapes(config, hook=None):
    matrix = Matrix.matrix_factory(config.decay)
    logging.info('processing data')
    for shape in config.shapes:
        shape = shape.map(config.projection.project)
        # TODO: skip shapes outside map extent
        shape.add_heat_to_matrix(matrix, config.kernel)
        if hook:
            hook(matrix)
    return matrix


def shapes_from_gpx(filename):
    track = TrackLog(filename)
    for trkseg in track.segments():
        for i, p1 in enumerate(trkseg[:-1]):
            p2 = trkseg[i + 1]
            yield LineSegment(p1.coords, p2.coords)


def shapes_from_file(filename):
    logging.info('reading points from %s' % filename)
    count = 0
    with open(filename, 'rU') as f:
        for line in f:
            line = line.strip()
            if len(line) > 0:  # ignore blank lines
                values = [float(x) for x in line.split()]
                assert len(values) == 2 or len(values) == 3, (
                    'input lines must have two or three values: %s' % line)
                (lat, lon) = values[0:2]
                weight = 1.0 if len(values) == 2 else values[2]
                count += 1
                yield Point(LatLon(lat, lon), weight)
        logging.info('read %d points' % count)


def shapes_from_csv(filename, ignore_csv_header):
    import csv
    logging.info('reading csv')
    count = 0
    with open(filename, 'rU') as f:
        reader = csv.reader(f)
        if ignore_csv_header:
            next(reader)  # Skip header line
        for row in reader:
            (lat, lon) = (float(row[0]), float(row[1]))
            count += 1
            yield Point(LatLon(lat, lon))
        logging.info('read %d points' % count)


def shapes_from_shp(filename):
    try:
        import ogr
    except ImportError:
        try:
            from osgeo import ogr
        except ImportError:
            raise ImportError("You need to have python-gdal bindings "
                              "installed")

    driver = ogr.GetDriverByName("ESRI Shapefile")
    dataSource = driver.Open(filename, 0)
    if dataSource is None:
        raise Exception("Not a valid shape file")

    layer = dataSource.GetLayer()
    if layer.GetGeomType() != 1:
        raise Exception("Only point layers are supported")

    spatial_reference = layer.GetSpatialRef()
    if spatial_reference is None:
        raise Exception("The shapefile doesn't have spatial reference")

    spatial_reference.AutoIdentifyEPSG()
    auth_code = spatial_reference.GetAuthorityCode(None)
    if auth_code == '':
        raise Exception("The input shapefile projection could not be "
                        "recognized")

    if auth_code != '4326':
        # TODO: implement reproject layer
        # (maybe geometry by geometry is easier)
        raise Exception("Currently only Lng-Lat WGS84 is supported "
                        "(EPSG 4326)")

    count = 0
    for feature in layer:
        geom = feature.GetGeometryRef()
        lat = geom.GetY()
        lon = geom.GetX()
        count += 1
        yield Point(LatLon(lat, lon))

    logging.info('read %d points' % count)


class Configuration(object):
    '''
    This object holds the settings for creating a heatmap as well as
    an iterator for the input data.

    Most of the command line processing is about settings and data, so
    the command line arguments are also processed with this object.
    This happens in two phases.

    First the settings are parsed and turned into more useful objects
    in set_from_options().  Command line flags go in, and the
    Configuration object is populated with the specified values and
    defaults.

    In the second phase, various other parameters are computed.  These
    are things we set automatically based on the other settings or on
    the data.  You can skip this if you set everything manually, but

    The idea is that someone could import this module, populate a
    Configuration instance manually, and run the process themselves.
    Where possible, this object contains instances, rather than option
    strings (e.g. for projection, kernel, colormap, etc).

    Every parameter is explained in the glossary dictionary, and only
    documented parameters are allowed.  Parameters default to None.
    '''

    glossary = {
        # Many of these are exactly the same as the command line option.
        # In those cases, the documentation is left blank.
        # Many have default values based on the command line defaults.
        'output': '',
        'width': '',
        'height': '',
        'margin': '',
        'shapes': 'unprojected iterable of shapes (Points and LineSegments)',
        'projection': 'Projection instance',
        'colormap': 'ColorMap instance',
        'decay': '',
        'kernel': 'kernel instance',
        'extent_in': 'extent in original space',
        'extent_out': 'extent in projected space',

        'background': '',
        'background_image': '',
        'background_brightness': '',

        # OpenStreetMap background tiles
        'osm': 'True/False; see command line options',
        'osm_base': '',
        'zoom': '',

        # These are for making an animation, ignored otherwise.
        'ffmpegopts': '',
        'keepframes': '',
        'frequency': '',
        'straggler_threshold': '',

        # We always instantiate an ArgumentParser in order to set up
        # default values.  You can use this ArgumentParser in your own
        # script, perhaps adding your own arguments.
        'argparser': 'ArgumentParser instance for command line processing',
    }

    _kernels = {'linear': LinearKernel,
                'gaussian': GaussianKernel, }
    _projections = {'equirectangular': EquirectangularProjection,
                    'mercator': MercatorProjection, }

    def __init__(self, use_defaults=True):
        for k in self.glossary.keys():
            setattr(self, k, None)
        self.argparser = self._make_argparser()
        if use_defaults:
            self.set_defaults()

    def set_defaults(self):
        args = self.argparser.parse_args([])
        self.set_from_options(args)

    def _make_argparser(self):
        '''Return a an ArgumentParser set up for our command line options.'''
        from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
        description = 'plot a heatmap from coordinate data'
        parser = ArgumentParser(description=description,
                                formatter_class=ArgumentDefaultsHelpFormatter)
        # TODO: allow multiple inputs of mixed types
        inputs = parser.add_mutually_exclusive_group()
        inputs.add_argument('-g', '--gpx', metavar='FILE')
        inputs.add_argument(
            '-p', '--points', metavar='FILE',
            help=(
                'File containing one space-separated coordinate pair per '
                'line, with optional point value as third term.'))
        inputs.add_argument(
            '--csv', metavar='FILE',
            help=(
                'File containing one comma-separated coordinate pair per '
                'line, the rest of the line is ignored.'))
        parser.add_argument(
            '--ignore_csv_header', action='store_true',
            help='Ignore first line of CSV input file.')
        parser.add_argument(
            '--shp_file', metavar='FILE',
            help=('ESRI Shapefile containing the points.'))
        parser.add_argument(
            '-s', '--scale', type=float,
            help='meters per pixel, approximate'),
        parser.add_argument(
            '-W', '--width', type=int,
            help='width of output image'),
        parser.add_argument(
            '-H', '--height', type=int,
            help='height of output image'),
        parser.add_argument(
            '-P', '--projection', metavar='NAME',
            choices=list(self._projections.keys()), default='mercator',
            help='choices: ' + ', '.join(self._projections.keys()) +
            '; default: %(default)s')
        parser.add_argument(
            '-e', '--extent', metavar='RANGE',
            help=(
                'Clip results to RANGE, which is specified as '
                'lat1,lon1,lat2,lon2;'
                ' (for square mercator: -85.0511,-180,85.0511,180)'))
        parser.add_argument(
            '-R', '--margin', type=int, default=0,
            help=(
                'Try to keep data at least this many pixels away from image '
                'border.'))
        parser.add_argument(
            '-r', '--radius', type=int, default=5,
            help='pixel radius of point blobs; default: %(default)s')
        parser.add_argument(
            '-d', '--decay', type=float, default=0.95,
            help=(
                'float in [0,1]; Larger values give more weight to data '
                'magnitude.  Smaller values are more democratic.  default:'
                '%(default)s'))
        parser.add_argument(
            '-S', '--save', metavar='FILE', help='save processed data to FILE')
        parser.add_argument(
            '-L', '--load', metavar='FILE',
            help='load processed data from FILE')
        parser.add_argument(
            '-o', '--output', metavar='FILE',
            help='name of output file (image or video)')
        parser.add_argument(
            '-a', '--animate', action='store_true',
            help='Make an animation instead of a static image')
        parser.add_argument(
            '--frequency', type=int, default=1,
            help='input points per animation frame; default: %(default)s')
        parser.add_argument(
            '--straggler_threshold', type=int, default=1,
            help='add one more animation frame if >= this many inputs remain')
        parser.add_argument(
            '-F', '--ffmpegopts', metavar='STR',
            help='extra options to pass to ffmpeg when making an animation')
        parser.add_argument(
            '-K', '--keepframes', action='store_true',
            help='keep intermediate images after creating an animation')
        parser.add_argument(
            '-b', '--background', metavar='COLOR',
            help='composite onto this background (color name or #rrggbb)')
        parser.add_argument(
            '-I', '--background_image', metavar='FILE',
            help='composite onto this image')
        parser.add_argument(
            '-B', '--background_brightness', type=float,
            help='Multiply each pixel in background image by this.')
        parser.add_argument(
            '-m', '--hsva_min', metavar='HEX',
            default=ColorMap.DEFAULT_HSVA_MIN_STR,
            help='hhhssvvaa hex for minimum data values; default: %(default)s')
        parser.add_argument(
            '-M', '--hsva_max', metavar='HEX',
            default=ColorMap.DEFAULT_HSVA_MAX_STR,
            help='hhhssvvaa hex for maximum data values; default: %(default)s')
        parser.add_argument(
            '-G', '--gradient', metavar='FILE',
            help=(
                'Take color gradient from this the first column of pixels in '
                'this image.  Overrides -m and -M.'))
        parser.add_argument(
            '-k', '--kernel',
            default='linear',
            choices=list(self._kernels.keys()),
            help=('Kernel to use for the falling-off function; choices: ' +
                  ', '.join(self._kernels.keys()) + '; default: %(default)s'))
        parser.add_argument(
            '--osm', action='store_true',
            help='Composite onto OpenStreetMap tiles')
        parser.add_argument(
            '--osm_base', metavar='URL',
            default='http://tile.openstreetmap.org',
            help='Base URL for map tiles; default %(default)s')
        parser.add_argument(
            '-z', '--zoom', type=int,
            help='Zoom level for OSM; 0 (the default) means autozoom')
        parser.add_argument('-v', '--verbose', action='store_true')
        parser.add_argument('--debug', action='store_true')
        return parser

    def set_from_options(self, options):
        for k in self.glossary.keys():
            try:
                setattr(self, k, getattr(options, k))
            except AttributeError:
                pass

        self.kernel = self._kernels[options.kernel](options.radius)
        self.projection = self._projections[options.projection]()

        if options.scale:
            self.projection.meters_per_pixel = options.scale

        if options.gradient:
            self.colormap = ColorMap(image=Image.open(options.gradient))
        else:
            self.colormap = ColorMap(
                hsva_min=ColorMap.str_to_hsva(options.hsva_min),
                hsva_max=ColorMap.str_to_hsva(options.hsva_max))

        if options.gpx:
            logging.debug('Reading from gpx: %s' % options.gpx)
            self.shapes = shapes_from_gpx(options.gpx)
        elif options.points:
            logging.debug('Reading from points: %s' % options.points)
            self.shapes = shapes_from_file(options.points)
        elif options.csv:
            logging.debug('Reading from csv: %s' % options.csv)
            self.shapes = shapes_from_csv(options.csv,
                                          options.ignore_csv_header)
        elif options.shp_file:
            logging.debug('Reading from Shape File: %s' % options.shp_file)
            self.shapes = shapes_from_shp(options.shp_file)

        if options.extent:
            (lat1, lon1, lat2, lon2) = \
                [float(f) for f in options.extent.split(',')]
            self.extent_in = Extent(coords=(LatLon(lat1, lon1),
                                            LatLon(lat2, lon2)))
        if options.background_image:
            self.background_image = Image.open(options.background_image)
            (self.width, self.height) = self.background_image.size

    def fill_missing(self):
        if not self.shapes:
            raise ValueError('no input specified')

        padding = self.margin + self.kernel.radius
        if not self.extent_in:
            logging.debug('reading input data')
            self.shapes = list(self.shapes)
            logging.debug('read %d shapes' % len(self.shapes))
            self.extent_in = Extent(shapes=self.shapes)

        if self.osm:
            get_osm_background(self, padding)
        else:
            if not self.projection.is_scaled():
                self.projection.auto_set_scale(self.extent_in, padding,
                                               self.width, self.height)
                if not (self.width or self.height or self.background_image):
                    raise ValueError('You must specify width or height or '
                                     'scale or background_image or both osm '
                                     'and zoom.')

        if self.background_brightness is not None:
            if self.background_image:
                self.background_image = self.background_image.point(
                    lambda x: x * self.background_brightness)
                self.background_brightness = None   # idempotence
            else:
                logging.warning(
                    'background brightness specified, but no background image')

        if not self.extent_out:
            self.extent_out = self.extent_in.map(self.projection.project)
            self.extent_out.grow(padding)
        logging.info('input extent: %s' % str(self.extent_out.map(
            self.projection.inverse_project)))
        logging.info('output extent: %s' % str(self.extent_out))


def main():
    logging.basicConfig(format='%(relativeCreated)8d ms  // %(message)s')
    config = Configuration(use_defaults=False)
    args = config.argparser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.load:
        logging.info('loading data')
        matrix = pickle.load(open(args.load, 'rb'))
        config, matrix['config'].argparser = matrix['config'], config.argparser
        del matrix['config']
        config.set_from_options(args)
        config.fill_missing()
    else:
        config.set_from_options(args)
        config.fill_missing()
        if args.animate:
            animator = ImageSeriesMaker(config)
            matrix = animator.run()
        else:
            matrix = process_shapes(config)
            matrix = matrix.finalized()

    if args.output and not args.animate:
        image = ImageMaker(config).make_image(matrix)
        image.save(args.output)

    if args.save:
        logging.info('saving data')
        matrix['config'] = config
        del config.argparser   # can't pickle an ArgumentParser
        pickle.dump(matrix, open(args.save, 'wb'), 2)

    logging.info('end')

if __name__ == '__main__':
    main()
