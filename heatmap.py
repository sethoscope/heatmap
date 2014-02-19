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
from time import mktime, strptime
from collections import defaultdict
import xml.etree.cElementTree as ET
from colorsys import hsv_to_rgb
try:
    import cPickle as pickle
except ImportError:
    import pickle

__version__ = '1.11'
options = None


class TrackLog:
    class Trkseg(list):  # for GPX <trkseg> tags
        pass

    class Trkpt:  # for GPX <trkpt> tags
        def __init__(self, lat, lon):
            self.coords = (float(lat), float(lon))

        def __str__(self):
            return '%f,%f' % self.coords

    def _Parse(self, filename):
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
                timestr = elem.findtext('time')
                if timestr:
                    timestr = timestr[:-1].split('.')[0] + ' GMT'
                    point.time = mktime(
                        strptime(timestr, '%Y-%m-%dT%H:%M:%S %Z'))
                elem.clear()  # clear the trkpt node to minimize memory usage

    def __init__(self, filename):
        self.filename = filename
    
    def segments(self):
        '''Parse file and yield segments containing points'''
        logging.info('reading GPX track from %s' % self.filename)
        return self._Parse(self.filename)


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

    def Project(self, coords):
        raise NotImplementedError

    def InverseProject(self, coords):   # Not all projections can support this.
        raise NotImplementedError

    def AutoSetScale(self, bounding_box_ll, padding, width=None, height=None):
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

        # xy coordinates are ints, so we'll work large
        # to minimize roundoff error.
        SCALE_FACTOR = 1000000.0
        self.pixels_per_degree = SCALE_FACTOR
        bounding_box_xy = bounding_box_ll.Map(self.Project)
        padding *= 2  # padding-per-edge -> padding-in-each-dimension
        if height:
            # TODO: div by zero error if all data exists at a single point.
            pixels_per_degree = pixels_per_lat = (
                float(height - padding) /
                bounding_box_xy.SizeY() * SCALE_FACTOR)
        if width:
            # TODO: div by zero error if all data exists at a single point.
            pixels_per_degree = (
                float(width - padding) /
                bounding_box_xy.SizeX() * SCALE_FACTOR)
            if height:
                pixels_per_degree = min(pixels_per_degree, pixels_per_lat)
        assert(pixels_per_degree > 0)
        self.pixels_per_degree = pixels_per_degree


# Treats Lat/Lon as a square grid.
class EquirectangularProjection(Projection):
    # http://en.wikipedia.org/wiki/Equirectangular_projection
    def Project(self, lat_lon):
        (lat, lon) = lat_lon
        x = int(lon * self.pixels_per_degree)
        y = -int(lat * self.pixels_per_degree)
        return (x, y)

    def InverseProject(self, x_y):
        (x, y) = x_y
        lat = -y / self.pixels_per_degree
        lon = x / self.pixels_per_degree
        return (lat, lon)


# If someone wants to use pixel coordinates instead of Lat/Lon, we
# could add an XYProjection.  EquirectangularProjection would work,
# but would be upside-down.

class MercatorProjection(Projection):
    def set_pixels_per_degree(self, val):
        super(MercatorProjection, self).set_pixels_per_degree(val)
        self._pixels_per_radian = val * (180 / math.pi)
    pixels_per_degree = property(Projection.get_pixels_per_degree,
                                 set_pixels_per_degree)

    def Project(self, lat_lon):
        (lat, lon) = lat_lon
        x = int(lon * self.pixels_per_degree)
        y = -int(self._pixels_per_radian * math.log(
            math.tan((math.pi/4 + math.pi/360 * lat))))
        return (x, y)

    def InverseProject(self, x_y):
        (x, y) = x_y
        lat = (360 / math.pi
               * math.atan(math.exp(-y / self._pixels_per_radian)) - 90)
        lon = x / self.pixels_per_degree
        return (lat, lon)

projections = {
    'equirectangular': EquirectangularProjection,
    'mercator': MercatorProjection,
}


class BoundingBox():
    '''This can be used for x,y or lat,lon; ints or floats.  It does not
    care which dimension is which, except that SizeX() and SizeY() refer
    to the first and second coordinate, regardless of which one is width
    and which is height.  (For Lat/Lon, SizeX() returns North/South
    extent.  This is confusing, but the alternative is to make assumptions
    based on whether the type (int or float) of the coordinates, which has
    too much hidden magic, or to let the caller set it in the constructor.
    Instead we just require you to know what you are doing.  There is a
    similar opportunity for magic with the desire to count fenceposts
    rather than distance, and here too we ignore the issue and let the
    caller deal with it as needed.'''
    def __init__(self, corners=None, shapes=None, coords=None):
        if corners:
            self.FromCorners(corners)
        elif shapes:
            self.FromShapes(shapes)
        elif coords:
            self.FromCoords(coords)
        else:
            raise ValueError('BoundingBox must be initialized')

    def __str__(self):
        return '%s,%s,%s,%s  (%sx%s)' % (
            self.minX, self.minY, self.maxX, self.maxY, self.SizeX(),
            self.SizeY())

    def Extent(self):
        return '%s,%s,%s,%s' % (self.minX, self.minY, self.maxX, self.maxY)

    def FromCorners(self, x1_y1_x2_y2):
        ((x1, y1), (x2, y2)) = x1_y1_x2_y2
        self.minX = min(x1, x2)
        self.minY = min(y1, y2)
        self.maxX = max(x1, x2)
        self.maxY = max(y1, y2)

    def FromShapes(self, shapes):
        if not shapes:
            return self.FromCorners(((0, 0), (0, 0)))
        # We loop through four times, but the code is nice and clean.
        self.minX = min(s.MinX() for s in shapes)
        self.maxX = max(s.MaxX() for s in shapes)
        self.minY = min(s.MinY() for s in shapes)
        self.maxY = max(s.MaxY() for s in shapes)

    def FromCoords(self, coords):
        if not coords:
            return self.FromCorners(((0, 0), (0, 0)))
        # We loop through four times, but the code is nice and clean.
        self.minX = min(c[0] for c in coords)
        self.maxX = max(c[0] for c in coords)
        self.minY = min(c[1] for c in coords)
        self.maxY = max(c[1] for c in coords)

    def Corners(self):
        return ((self.minX, self.minY), (self.maxX, self.maxY))

    # We use "SixeX" and "SizeY" instead of Width and Height because we
    # use these both for XY and LatLon, and they're in opposite order.
    # Rather than have the object try to keep track, we just choose not
    # to need it.  In a strongly typed language, we could distinguish
    # between degrees and pixels.  We could do that here by overloading
    # floats and ints, but that would just be a different kind of
    # confusion and probably easier to make mistakes with.
    def SizeX(self):
        return self.maxX - self.minX

    def SizeY(self):
        return self.maxY - self.minY

    def Grow(self, pad):
        self.minX -= pad
        self.minY -= pad
        self.maxX += pad
        self.maxY += pad

    def ClipToSize(self, width=None, height=None, include_fenceposts=True):
        fencepost = include_fenceposts and 1 or 0
        if width:
            current_width = self.SizeX()
            # round up
            self.maxX += int(float(1 + width - current_width - fencepost) / 2)
            self.minX = self.maxX - width + fencepost

        if height:
            current_height = self.SizeY()
            # round up
            self.maxY += int(
                float(1 + height - current_height - fencepost) / 2)
            self.minY = self.maxY - height + fencepost

    def IsInside(self, x_y):
        (x, y) = x_y
        return (
            x >= self.minX and x <= self.maxX
            and y >= self.minY and y <= self.maxY)

    def Map(self, func):
        '''Returns a new BoundingBox whose corners are a function of the
        corners of this one.  The expected use is to project a BoundingBox
        onto a map.  For example: bbox_xy = bbox_ll.Map(projector.Project)'''
        return BoundingBox(
            corners=(func((self.minX, self.minY)),
                     func((self.maxX, self.maxY))))


class Matrix(defaultdict):
    '''An abstract sparse matrix, with data stored as {coord : value}.'''

    @staticmethod
    def MatrixFactory(decay):
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

    def Add(self, coord, val):
        raise NotImplementedError

    def BoundingBox(self):
        return(BoundingBox(coords=self.keys()))

    def Finalized(self):
        return self


class SummingMatrix(Matrix):
    def Add(self, coord, val):
        self[coord] += val


class MaxingMatrix(Matrix):
    def Add(self, coord, val):
        self[coord] = max(val, self.get(coord, val))


class AppendingMatrix(Matrix):
    def __init__(self, decay):
        self.default_factory = list
        self.decay = decay

    def Add(self, coord, val):
        self[coord].append(val)

    def Finalized(self):
        logging.info('combining coincident points')
        dr = DiminishingReducer(self.decay)
        m = Matrix()
        for (coord, values) in self.items():
            m[coord] = dr.Reduce(values)
        return m


class DiminishingReducer():
    def __init__(self, decay):
        '''This reducer returns a weighted sum of the values, where weight
        N is pow(decay,N).  This means the largest value counts fully, but
        additional values have diminishing contributions.  decay=0.0 makes
        the reduction equivalent to max(), which makes each data point
        visible, but says nothing about their relative magnitude.
        decay=1.0 makes this like sum(), which makes the relative magnitude
        of the points more visible, but could make smaller values hard to see.
        Experiment with values between 0 and 1.  Values outside that range
        will give weird results.'''
        self.decay = decay

    def Reduce(self, values):
        # It would be nice to do this on the fly, while accumulating data, but
        # it needs to be insensitive to data order.
        weight = 1.0
        total = 0.0
        values.sort(reverse=True)
        for value in values:
            total += value * weight
            weight *= self.decay
        return total


class Point:
    def __init__(self, x_y, weight=1.0):
        (x, y) = x_y
        self.x = x
        self.y = y
        self.weight = weight

    def __str__(self):
        return 'P(%s,%s)' % (self.x, self.y)

    @staticmethod
    def GeneralDistance(x, y):
        # assumes square units, which causes distortion in some projections
        return (x ** 2 + y ** 2) ** 0.5

    def Distance(self, x_y):
        (x, y) = x_y
        return self.GeneralDistance(self.x - x, self.y - y)

    def MinX(self):
        return self.x

    def MaxX(self):
        return self.x

    def MinY(self):
        return self.y

    def MaxY(self):
        return self.y

    # From a modularity standpoint, it would be reasonable to cache
    # distances, not heat values, and let the kernel cache the
    # distance to heat map, but this is substantially faster.
    heat_cache = {}
    @classmethod
    def InitializeHeatCache(cls, kernel):
        cache = {}
        for x in range(kernel.radius + 1):
            for y in range(kernel.radius + 1):
                cache[(x, y)] = kernel.Heat(cls.GeneralDistance(x, y))
        cls.heat_cache[kernel] = cache

    def AddHeatToMatrix(self, matrix, kernel):
        if kernel not in Point.heat_cache:
            Point.InitializeHeatCache(kernel)
        cache = Point.heat_cache[kernel]
        for dx in range(-kernel.radius, kernel.radius + 1):
            for dy in range(-kernel.radius, kernel.radius + 1):
                matrix.Add((self.x + dx, self.y + dy),
                           self.weight * cache[(abs(dx), abs(dy))])

    def Map(self, func):
        return Point(func((self.x, self.y)), self.weight)


class LineSegment:
    def __init__(self, x1_y1, x2_y2, weight=1.0):
        (x1, y1) = x1_y1
        (x2, y2) = x2_y2
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.weight = weight
        self.length_squared = float(
            (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1))

    def __str__(self):
        return 'LineSegment((%s,%s), (%s,%s))' % (
            self.x1, self.y1, self.x2, self.y2)

    def MinX(self):
        return min(self.x1, self.x2)

    def MaxX(self):
        return max(self.x1, self.x2)

    def MinY(self):
        return min(self.y1, self.y2)

    def MaxY(self):
        return max(self.y1, self.y2)

    def Distance(self, x_y):
        # http://stackoverflow.com/questions/849211/shortest-distance-between-a-point-and-a-line-segment
        # http://www.topcoder.com/tc?d1=tutorials&d2=geometry1&module=Static#line_point_distance
        # http://local.wasp.uwa.edu.au/~pbourke/geometry/pointline/
        (x, y) = x_y
        try:
            dx = (self.x2 - self.x1)
            dy = (self.y2 - self.y1)
            u = ((x - self.x1) * dx + (y - self.y1) * dy) / self.length_squared
            if u < 0:
                u = 0
            elif u > 1:
                u = 1
        except ZeroDivisionError:
            u = 0  # Our line is zero-length.  That's ok.
        dx = self.x1 + u * dx - x
        dy = self.y1 + u * dy - y
        return math.sqrt(dx * dx + dy * dy)

    def AddHeatToMatrix(self, matrix, kernel):
        # Iterate over every point in a bounding box around this, with an
        # extra margin given by the kernel's self-reported maximum range.
        # TODO: There is probably a more clever iteration that skips more
        # of the empty space.
        for x in range(self.MinX() - kernel.radius,
                       self.MaxX() + kernel.radius + 1):
            for y in range(self.MinY() - kernel.radius,
                           self.MaxY() + kernel.radius + 1):
                heat = kernel.Heat(self.Distance((x, y)))
                if heat:
                    matrix.Add((x, y), self.weight * heat)

    def Map(self, func):
        xy1 = func((self.x1, self.y1))
        xy2 = func((self.x2, self.y2))
        # Quantizing can make both endpoints the same, turning the
        # LineSegment into an inefficient Point.  Better to replace it.
        if xy1 == xy2:
            return Point(xy1, self.weight)
        else:
            return LineSegment(xy1, xy2, self.weight)


class LinearKernel:
    '''Uses a linear falloff, essentially turning a point into a cone.'''
    def __init__(self, radius):
        self.radius = radius  # in pixels
        self.radius_float = float(radius)  # worthwhile time saver

    def Heat(self, distance):
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

    def Heat(self, distance):
        '''Returns 1.0 at center, 1/e at radius pixels from center.'''
        return math.e ** (-distance * self.scale)


kernels = {
    'linear': LinearKernel,
    'gaussian': GaussianKernel,
}


class ColorMap:
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
            string = string[1:]  # Leading "#" was once required, is now optional.
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
        self.values = []
        if hsva_min:
            assert hsva_max is not None
            # Turn (h1,s1,v1,a1), (h2,s2,v2,a2) into (h2-h1,s2-s1,v2-v1,a2-a1)
            hsva_range = list(map(lambda min, max: max - min, hsva_min, hsva_max))
            for value in range(0, steps):
                hsva = list(map(
                    lambda range, min: value / float(steps - 1) * range + min,
                    hsva_range, hsva_min))
                hsva[0] = hsva[0] % 1  # in case hue is out of range
                rgba = tuple(
                    [int(x * 255) for x in hsv_to_rgb(*hsva[0:3]) + (hsva[3],)])
                self.values.append(rgba)
        else:
            assert image is not None
            assert image.mode == 'RGBA', (
                'Gradient image must be RGBA.  Yours is %s.' % image.mode)
            num_rows = image.size[1]
            self.values = [image.getpixel((0, row)) for row in range(num_rows)]
            self.values.reverse()

    def get(self, floatval):
        return self.values[int(floatval * (len(self.values) - 1))]


class ImageMaker():
    def __init__(self, colormap, background=None, background_image=None):
        '''Each argument to the constructor should be a 4-tuple of (hue,
        saturaton, value, alpha), one to use for minimum data values and
        one for maximum.  Each should be in [0,1], however because hue is
        circular, you may specify hue in any range and it will be shifted
        into [0,1] as needed.  This is so you can wrap around the color
        wheel in either direction.'''
        self.colormap = colormap
        self.background_image = background_image
        self.background = None
        if background and not background_image:
            self.background = ImageColor.getrgb(background)

    @staticmethod
    def _blend_pixels(a, b):
        # a is RGBA, b is RGB; we could write this more generically,
        # but why complicate things?
        alpha = a[3] / 255.0
        return tuple(
            map(lambda aa, bb: int(aa * alpha + bb * (1 - alpha)), a[:3], b))


    def SavePNG(
            self, matrix, filename, requested_width=None,
            requested_height=None, bounding_box=None):
        if not bounding_box:
            bounding_box = matrix.BoundingBox()
        bounding_box.ClipToSize(requested_width, requested_height)
        ((minX, minY), (maxX, maxY)) = bounding_box.Corners()
        width = maxX - minX + 1
        height = maxY - minY + 1
        logging.info('saving image (%d x %d)' % (width, height))

        if self.background:
            img = Image.new('RGB', (width, height), self.background)
        else:
            img = Image.new('RGBA', (width, height))

        maxval = max(matrix.values())
        pixels = img.load()

        # Iterating just over the non-zero data points is ideal when
        # plotting the whole image, but for generating tile sets, it might
        # make more sense for the caller to partition the points and pass in
        # a list of points to use for each image.  That way we only iterate
        # over the points once, rather than once per image.  That also gives
        # the caller an opportunity to do something better for tiles that
        # contain no data.
        for ((x, y), val) in matrix.items():
            if bounding_box.IsInside((x, y)):
                if self.background:
                    pixels[x - minX, y - minY] = ImageMaker._blend_pixels(
                        self.colormap.get(val / maxval),
                        self.background)
                else:
                    pixels[x - minX, y - minY] = self.colormap.get(val / maxval)
        if self.background_image:
            # Is this really the best way?
            img = Image.composite(img, self.background_image, img.split()[3])
        img.save(filename)


class ImageSeriesMaker():
    '''Creates a movie showing the data appearing on the heatmap.'''
    def __init__(self, config):
        self.config = config
        self.image_maker = ImageMaker(config.colormap, config.background, config.background_image)
        self.tmpdir = tempfile.mkdtemp()
        self.imgfile_template = os.path.join(self.tmpdir, 'frame-%05d.png')


    def _SaveImage(self, matrix):
        self.frame_count += 1
        logging.info('Frame %d' % (self.frame_count))
        matrix = matrix.Finalized()
        self.image_maker.SavePNG(
            matrix, self.imgfile_template % self.frame_count,
            self.config.width, self.config.height, self.config.bounding_box_xy)

    def MaybeSaveImage(self, matrix):
        self.inputs_since_output += 1
        if self.inputs_since_output >= self.config.frequency:
            self._SaveImage(matrix)
            self.inputs_since_output = 0

    @staticmethod
    def CreateMovie(infiles, outfile, ffmpegopts):
        command = ['ffmpeg', '-i', infiles]
        if ffmpegopts:
            # I hope they don't have spaces in their arguments
            command.extend(ffmpegopts.split())
        command.append(outfile)
        logging.info('Encoding video: %s' % ' '.join(command))
        subprocess.call(command)


    def MainLoop(self):
        logging.info('Putting animation frames in %s' % self.tmpdir)
        self.inputs_since_output = 0
        self.frame_count = 0
        matrix = ProcessShapes(self.config, self.MaybeSaveImage)
        if ( not self.frame_count
             or self.inputs_since_output >= self.config.straggler_threshold ):
            self._SaveImage(matrix)
        self.CreateMovie(self.imgfile_template,
                         self.config.output,
                         self.config.ffmpegopts)
        if self.config.keepframes:
            logging.info('The animation frames are in %s' % self.tmpdir)
        else:
            shutil.rmtree(self.tmpdir)
        return matrix


def _GetOSMImage(bbox, zoom, osm_base):
    # Just a wrapper for osm.createOSMImage to translate coordinate schemes
    try:
        from osmviz.manager import PILImageManager, OSMManager
        osm = OSMManager(
            image_manager=PILImageManager('RGB'),
            server=osm_base)
        ((lat1, lon1), (lat2, lon2)) = bbox.Corners()
        image, bounds = osm.createOSMImage((lat1, lat2, lon1, lon2), zoom)
        (lat1, lat2, lon1, lon2) = bounds
        return image, BoundingBox(corners=((lat1, lon1), (lat2, lon2)))
    except ImportError as e:
        logging.error(
            "ImportError: %s.\n"
            "The --osm option depends on the osmviz module, available from\n"
            "http://cbick.github.com/osmviz/\n\n" % str(e))
        sys.exit(1)


def _ScaleForOSMZoom(zoom):
    return 256 * pow(2, zoom) / 360.0


def ChooseOSMZoom(bbox_ll, padding):
    # Since we know we're only going to do this with Mercator, we could do
    # a bit more math and solve this directly, but as a first pass method,
    # we instead project the bounding box into pixel-land at a high zoom
    # level, then see the power of two we're off by.
    if options.zoom:
        return options.zoom
    crazy_zoom_level = 30
    proj = MercatorProjection()
    scale = _ScaleForOSMZoom(crazy_zoom_level)
    proj.pixels_per_degree = scale
    bbox_crazy_xy = bbox_ll.Map(proj.Project)
    if options.width:
        size_ratio = width_ratio = (
            float(bbox_crazy_xy.SizeX()) / (options.width - 2 * padding))
    if options.height:
        size_ratio = (
            float(bbox_crazy_xy.SizeY()) / (options.height - 2 * padding))
        if options.width:
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


def GetOSMBackground(bbox_ll, padding, osm_base):
    zoom = ChooseOSMZoom(bbox_ll, padding)
    proj = MercatorProjection()
    proj.pixels_per_degree = _ScaleForOSMZoom(zoom)
    bbox_xy = bbox_ll.Map(proj.Project)
    # We're not checking that the padding fits within the specified size.
    bbox_xy.Grow(padding)
    bbox_ll = bbox_xy.Map(proj.InverseProject)
    image, img_bbox_ll = _GetOSMImage(bbox_ll, zoom, osm_base)
    img_bbox_xy = img_bbox_ll.Map(proj.Project)

    # TODO: this crops to our data extent, which means we're not making
    # an image of the requested dimensions.  Perhaps we should let the
    # user specify whether to treat the requested size as min,max,exact.
    (x_offset, y_offset) = map(
        lambda a, b: a - b, bbox_xy.Corners()[0], img_bbox_xy.Corners()[0])
    x_size = bbox_xy.SizeX() + 1
    y_size = bbox_xy.SizeY() + 1
    image = image.crop((
        x_offset,
        y_offset,
        x_offset + x_size,
        y_offset + y_size))
    return image, bbox_ll, proj


def ProcessShapes(config, hook=None):
    matrix = Matrix.MatrixFactory(config.decay)
    logging.info('processing data')
    for shape in config.shapes:
        shape = shape.Map(config.projection.Project)
        # TODO: skip shapes outside map extent
        shape.AddHeatToMatrix(matrix, config.kernel)
        if hook:
            hook(matrix)
    return matrix


def setup_cmdline_options():
    '''Return a an OptionParser set up for our command line options.'''
    # TODO: convert to argparse
    from optparse import OptionParser
    optparser = OptionParser(version=__version__)
    optparser.add_option('-g', '--gpx', metavar='FILE')
    optparser.add_option(
        '-p', '--points', metavar='FILE',
        help=(
            'File containing one space-separated coordinate pair per line, '
            'with optional point value as third term.'))
    optparser.add_option(
        '', '--csv', metavar='FILE',
        help=(
            'File containing one comma-separated coordinate pair per line, '
            'the rest of the line is ignored.'))
    optparser.add_option(
        '', '--ignore_csv_header', action='store_true',
        help='Ignore first line of CSV input file.')

    optparser.add_option(
        '-s', '--scale', metavar='FLOAT', type='float',
        help='meters per pixel, approximate'),
    optparser.add_option(
        '-W', '--width', metavar='INT', type='int',
        help='width of output image'),
    optparser.add_option(
        '-H', '--height', metavar='INT', type='int',
        help='height of output image'),
    optparser.add_option(
        '-P', '--projection', metavar='NAME', type='choice',
        choices=list(projections.keys()), default='mercator',
        help='choices: ' + ', '.join(projections.keys()) +
        '; default: %default')
    optparser.add_option(
        '-e', '--extent', metavar='RANGE',
        help=(
            'Clip results to RANGE, which is specified as lat1,lon1,lat2,lon2;'
            ' (for square mercator: -85.0511,-180,85.0511,180)'))
    optparser.add_option(
        '-R', '--margin', metavar='INT', type='int', default=0,
        help=(
            'Try to keep data at least this many pixels away from image '
            'border.'))
    optparser.add_option(
        '-r', '--radius', metavar='INT', type='int', default=15,
        help='pixel radius of point blobs; default: %default')
    optparser.add_option(
        '-d', '--decay', metavar='FLOAT', type='float', default=0.95,
        help=(
            'float in [0,1]; Larger values give more weight to data '
            'magnitude.  Smaller values are more democratic.  default:'
            '%default'))
    optparser.add_option(
        '-S', '--save', metavar='FILE', help='save processed data to FILE')
    optparser.add_option(
        '-L', '--load', metavar='FILE', help='load processed data from FILE')
    optparser.add_option(
        '-o', '--output', metavar='FILE',
        help='name of output file (image or video)')
    optparser.add_option(
        '-a', '--animate', action='store_true',
        help='Make an animation instead of a static image')
    optparser.add_option(
        '', '--frequency', type='int', default=1,
        help='input points per animation frame; default: %default')
    optparser.add_option(
        '', '--straggler_threshold', type='int', default=1,
        help='add one more animation frame if >= this many inputs remain')
    optparser.add_option(
        '-F', '--ffmpegopts', metavar='STR',
        help='extra options to pass to ffmpeg when making an animation')
    optparser.add_option(
        '-K', '--keepframes', action='store_true',
        help='keep intermediate images after creating an animation')
    optparser.add_option(
        '-b', '--background', metavar='COLOR',
        help='composite onto this background (color name or #rrggbb)')
    optparser.add_option(
        '-I', '--background_image', metavar='FILE',
        help='composite onto this image')
    optparser.add_option(
        '-B', '--background_brightness', type='float', metavar='NUM',
        help='Multiply each pixel in background image by this.')
    optparser.add_option(
        '-m', '--hsva_min', metavar='HEX', default='000ffff00',
        help='hhhssvvaa hex for minimum data values; default: %default')
    optparser.add_option(
        '-M', '--hsva_max', metavar='HEX', default='02affffff',
        help='hhhssvvaa hex for maximum data values; default: %default')
    optparser.add_option(
        '-G', '--gradient', metavar='FILE',
        help=(
        'Take color gradient from this the first column of pixels in '
        'this image.  Overrides -m and -M.'))
    optparser.add_option(
        '-k', '--kernel',
        type='choice',
        default='linear',
        choices=list(kernels.keys()),
        help=('Kernel to use for the falling-off function; choices: ' +
              ', '.join(kernels.keys()) + '; default: %default'))
    optparser.add_option(
        '', '--osm', action='store_true',
        help='Composite onto OpenStreetMap tiles')
    optparser.add_option(
        '', '--osm_base', metavar='URL',
        default='http://tile.openstreetmap.org',
        help='Base URL for map tiles; default %default')
    optparser.add_option(
        '-z', '--zoom', type='int',
        help='Zoom level for OSM; 0 (the default) means autozoom')
    optparser.add_option('-v', '--verbose', action='store_true')
    optparser.add_option('', '--debug', action='store_true')
    return optparser


class Configuration(object):
    '''
    There are lots of configuration parameters, used at various levels
    of the module.  To simplify passing them down to where they are used
    without cluttering up the code in between, we stash everything in
    this object.

    Most of the command line processing is done here in from_options().
    The idea is that someone could import this module, populate a
    Configuration instance manually, and run the process themselves.
    Where possible, this object contains instances, rather than option
    strings (e.g. for projection, kernel, colormap, etc).

    Every parameter is explained in the glossary dictionary, and only
    documented parameters are allowed.  Parameters default to None.
    '''

    glossary = {
        'output' : 'output filename',
        'width' : 'width of output image',
        'height' : 'height of output image',
        'shapes' : 'unprojected list of shapes (Points and LineSegments)',
        'projection' : 'Projection instance',
        'colormap' : 'ColorMap instance',
        'decay' : 'see command line options',
        'kernel' : 'kernel instance',
        'bounding_box_xy' : 'optional extent in projected space',
        # TODO: add optional bounding_box_ll for filtering input
        'background': 'composite onto this background color',
        'background_image': 'composite onto this image',

        # These are for making an animation, ignored otherwise.
        'ffmpegopts' : 'extra options to pass to ffmpeg (for animations)',
        'keepframes' : 'whether to keep image frames after creating animation',
        'frequency'  : 'number of input shapes per frame',
        'straggler_threshold' : 'add last frame if >= this many inputs remain',

        # If you need new slots with which to pass data to your own
        # hacks, you'll need to add to this glossary.  You can do that
        # at run time if that's more convenient.
        }
    __slots__ = glossary.keys()

    def __init__(self):
        for k in self.glossary.keys():
            setattr(self, k, None)  # everything defaults to None

    def from_options(self, options):
        for k in ('width', 'height',
                  'keepframes',
                  'ffmpegopts',
                  'frequency',
                  'straggler_threshold',
                  'output',
                  'decay',
                  'background',
                  ):
            setattr(self, k, getattr(options, k))

        if options.gpx:
            logging.debug('Reading from gpx: %s' + options.gpx)
            self.shapes = shapes_from_gpx(options.gpx)
        elif options.points:
            logging.debug('Reading from points: %s' + options.points)
            self.shapes = shapes_from_file(options.points)
        elif options.csv:
            logging.debug('Reading from csv: %s' + options.csv)
            self.shapes = shapes_from_csv(options.csv, options.ignore_csv_header)
        else:
            raise ValueError('no input file')

        if ((options.gpx or options.points or options.csv)
            and not ((options.width or options.height or options.scale
                      or options.background_image)
                     or (options.osm and options.zoom))):
            raise ValueError(
                "With --gpx, --points or --csv, you must also specify at least "
                "one of --width, --height,\n --scale, or --background_image, or "
                "both --osm and --zoom.")

        self.kernel = kernels[options.kernel](options.radius)

        if options.gradient:
            self.colormap = ColorMap(image = Image.open(options.gradient))
        else:
            self.colormap = ColorMap(hsva_min = ColorMap.str_to_hsva(options.hsva_min),
                                     hsva_max = ColorMap.str_to_hsva(options.hsva_max))

        bounding_box_ll = None
        bounding_box_xy_padding = options.margin
        if options.extent:
            (lat1, lon1, lat2, lon2) = [float(f) for f in options.extent.split(',')]
            bounding_box_ll = BoundingBox(corners=((lat1, lon1), (lat2, lon2)))
        else:
            self.shapes = list(self.shapes)
            logging.debug('num shapes: %d' % len(self.shapes))
            bounding_box_ll = BoundingBox(shapes=self.shapes)
            bounding_box_xy_padding += options.radius

        # background image
        if options.background_image:
            self.background_image = Image.open(options.background_image)
            (self.width, self.height) = background_image.size

        if options.osm:
            (self.background_image,
             bounding_box_ll,
             self.projection) = GetOSMBackground(bounding_box_ll,
                                                 bounding_box_xy_padding,
                                                 options.osm_base)
            (self.width, self.height) = background_image.size
            bounding_box_xy_padding = 0  # already baked in

        if options.background_brightness:
            if self.background_image:
                self.background_image = background_image.point(
                    lambda x: x * options.background_brightness)
            else:
                logging.warning(
                    'background brightness specified, but no background image')

        if not self.projection:
            self.projection = projections[options.projection]()
            if options.scale:
                self.projection.meters_per_pixel = options.scale
            else:
                self.projection.AutoSetScale(bounding_box_ll,
                                             bounding_box_xy_padding,
                                             self.width, self.height)
        self.bounding_box_xy = bounding_box_ll.Map(self.projection.Project)
        self.bounding_box_xy.Grow(bounding_box_xy_padding)
        if not options.extent:
            logging.info('Map extent: %s' % self.bounding_box_xy.Map(
                self.projection.InverseProject).Extent())


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
                yield Point((lat, lon), weight)
        logging.info('read %d points' % count)

def shapes_from_csv(filename, ignore_csv_header):
    import csv
    logging.info('reading csv')
    count = 0
    with open(filename, 'ru') as f:
        reader = csv.reader(f)
        if ignore_csv_header:
            reader.next()  # Skip header line
        for row in reader:
            (lat, lon) = (float(row[0]), float(row[1]))
            count += 1
            yield Point((lat, lon))
        logging.info('read %d points' % count)


def main():
    logging.basicConfig(format='%(relativeCreated)8d ms  // %(message)s')
    optparser = setup_cmdline_options()
    (options, args) = optparser.parse_args()

    if options.verbose:
        logging.getLogger().setLevel(logging.INFO)
    if options.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    if options.load:
        logging.info('loading data')
        matrix = pickle.load(open(options.load, 'rb'))
        config = matrix['config']
        del matrix['config']
    else:
        config = Configuration()
        config.from_options(options)
        if options.animate:
            animator = ImageSeriesMaker(config)
            matrix = animator.MainLoop()
        else:
            matrix = ProcessShapes(config)
            matrix = matrix.Finalized()

    if options.output and not options.animate:
        ImageMaker(config.colormap, config.background, config.background_image).SavePNG(
            matrix, config.output, config.width, config.height,
            config.bounding_box_xy)

    if options.save:
        logging.info('saving data')
        matrix['config'] = config
        pickle.dump(matrix, open(options.save, 'wb'), 2)

    logging.info('end')

if __name__ == '__main__':
    main()
