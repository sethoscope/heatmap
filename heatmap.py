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
import Image
import ImageColor
from time import mktime, strptime
from collections import defaultdict
import xml.etree.cElementTree as ET
from colorsys import hsv_to_rgb

__version__ = '1.10'
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
        for event, elem in ET.iterparse(filename, ('start', 'end')):
            elem.tag = elem.tag[elem.tag.rfind('}') + 1:]   # remove namespace
            if elem.tag == "trkseg":
                if event == 'start':
                    self.segments.append(TrackLog.Trkseg())
                else:  # event == 'end'
                    elem.clear()  # delete contents from parse tree
            elif elem.tag == 'trkpt' and event == 'end':
                point = TrackLog.Trkpt(elem.attrib['lat'], elem.attrib['lon'])
                self.segments[-1].append(point)
                timestr = elem.findtext('time')
                if timestr:
                    timestr = timestr[:-1].split('.')[0] + ' GMT'
                    point.time = mktime(
                        strptime(timestr, '%Y-%m-%dT%H:%M:%S %Z'))
                elem.clear()  # clear the trkpt node to minimize memory usage

    def __init__(self, filename):
        self.segments = []
        logging.info('reading GPX track from %s' % filename)
        self._Parse(filename)
        logging.info('track length: %d points in %d segments'
                     % (sum(len(seg) for seg in self.segments),
                        len(self.segments)))


class Projection():
    def SetScale(self, pixels_per_degree):
        raise NotImplementedError

    def Project(self, coords):
        raise NotImplementedError

    def InverseProject(self, coords):   # Not all projections can support this.
        raise NotImplementedError

    def AutoSetScale(self, bounding_box_ll, padding):
        if options.scale:
            # Here we assume the Earth is a sphere of radius 6378137m.
            # earth circumference @ equator is roughly 40075017 meters
            # (in WGS-84)
            # so meters per degree longitude at equator =~ 111319.5
            # px/deg = m/deg * px/m
            pixels_per_degree = 111319.5 / options.scale
        else:
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
            self.SetScale(SCALE_FACTOR)
            bounding_box_xy = bounding_box_ll.Map(self.Project)
            padding *= 2  # padding-per-edge -> padding-in-each-dimension
            if options.height:
                # TODO: div by zero error if all data exists at a single point.
                pixels_per_degree = pixels_per_lat = (
                    float(options.height - padding) /
                    bounding_box_xy.SizeY() * SCALE_FACTOR)
            if options.width:
                # TODO: div by zero error if all data exists at a single point.
                pixels_per_degree = (
                    float(options.width - padding) /
                    bounding_box_xy.SizeX() * SCALE_FACTOR)
                if options.height:
                    pixels_per_degree = min(pixels_per_degree, pixels_per_lat)
        assert(pixels_per_degree > 0)
        self.SetScale(pixels_per_degree)
        logging.info('Scale: %f' % (111319.5 / pixels_per_degree))


# Treats Lat/Lon as a square grid.
class EquirectangularProjection(Projection):
    # http://en.wikipedia.org/wiki/Equirectangular_projection
    def SetScale(self, pixels_per_degree):
        self.pixels_per_degree = pixels_per_degree

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
    def SetScale(self, pixels_per_degree):
        self.pixels_per_degree = pixels_per_degree
        self.pixels_per_radian = pixels_per_degree * (180 / math.pi)

    def Project(self, lat_lon):
        (lat, lon) = lat_lon
        x = int(lon * self.pixels_per_degree)
        y = -int(self.pixels_per_radian * math.log(
            math.tan((math.pi/4 + math.pi/360 * lat))))
        return (x, y)

    def InverseProject(self, x_y):
        (x, y) = x_y
        lat = (
            360 / math.pi * math.atan(
                math.exp(-y / self.pixels_per_radian)) - 90)
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
    def __init__(self, corners=None, shapes=None, string=None):
        if corners:
            self.FromCorners(corners)
        elif shapes:
            self.FromShapes(shapes)
        elif string:
            (lat1, lon1, lat2, lon2) = [float(f) for f in string.split(',')]
            self.FromCorners(((lat1, lon1), (lat2, lon2)))
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

    def Corners(self):
        return ((self.minX, self.minY), (self.maxX, self.maxY))

    # We use "SixeX" and "SizeY" instead of Width and Height because we
    # use these both for XY and LatLon, and they're in opposite order.
    # Rather than have the object try to keep track, we just choose not
    # to need it.  In a strongly typed language, we'd could distinguish
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


class Matrix:
    @classmethod
    def MatrixFactory(cls, decay):
        # If decay is 0 or 1, we can accumulate as we go and save lots of
        # memory.
        if decay == 1.0:
            logging.info('creating a summing matrix')
            return SummingMatrix()
        elif decay == 0.0:
            logging.info('creating a maxing matrix')
            return MaxingMatrix()
        logging.info('creating an appending matrix')
        return AppendingMatrix()

    def __init__(self):
        self.data = {}  # sparse matrix, stored as {(x,y) : value}

    def Add(self, coord, val, adder=lambda x, y: x + y):
        raise NotImplementedError

    def Set(self, coord, val):
        self.data[coord] = val

    def iteritems(self):
        return self.data.items()

    def Max(self):
        return max(self.data.values())

    def items(self):
        return self.data.items()

    def Get(self, coord):
        return self.data[coord]   # will throw KeyError for unset coord

    def BoundingBox(self):
        return(BoundingBox(iter=self.data.iterkeys()))

    def Finalized(self):
        return self


class SummingMatrix(Matrix):
    def Add(self, coord, val):
        self.data[coord] = val + self.data.get(coord, 0.0)


class MaxingMatrix(Matrix):
    def Add(self, coord, val):
        self.data[coord] = max(val, self.data.get(coord, val))


class AppendingMatrix(Matrix):
    def __init__(self):
        self.data = defaultdict(list)

    def Add(self, coord, val):
        self.data[coord].append(val)

    def Finalized(self):
        logging.info('combining coincident points')
        dr = DiminishingReducer(options.decay)
        m = Matrix()
        for (coord, values) in self.iteritems():
            m.Set(coord, dr.Reduce(values))
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


def str2hsva(string):
    'Turns 06688bbff into (102, 136, 187, 255); the first number is 3 digits!'
    if string.startswith('#'):
        string = string[1:]  # Leading "#" was once required, is now optional.
    return (int(string[0:3], 16),
            int(string[3:5], 16),
            int(string[5:7], 16),
            int(string[7:9], 16))


class ColorMap:
    def __getitem__(self, i):
        return self.data[i]

    def FromHsvaRangeStrings(self, hsva_min_str, hsva_max_str):
        hsva_min = [_8bitInt_to_float(x) for x in str2hsva(hsva_min_str)]
        hsva_max = [_8bitInt_to_float(x) for x in str2hsva(hsva_max_str)]
        # more useful this way
        hsva_range = list(map(lambda min, max: max - min, hsva_min, hsva_max))
        self.data = []
        for value in range(0, 256):
            hsva = list(map(
                lambda range, min: value / 255.0 * range + min,
                hsva_range, hsva_min))
            hsva[0] = hsva[0] % 1  # in case hue is out of range
            rgba = tuple(
                [int(x * 255) for x in hsv_to_rgb(*hsva[0:3]) + (hsva[3],)])
            self.data.append(rgba)

    def FromImage(self, img):
        assert img.mode == 'RGBA', (
            'Gradient image must be RGBA.  Yours is %s.' % img.mode)
        maxY = img.size[1] - 1
        self.data = []
        for value in range(256):
            self.data.append(img.getpixel((0, maxY * (255 - value) / 255)))


def _blend_pixels(a, b):
    # a is RGBA, b is RGB; we could write this more generically,
    # but why complicate things?
    alpha = a[3] / 255.0
    return tuple(
        map(lambda aa, bb: int(aa * alpha + bb * (1 - alpha)), a[:3], b))


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

        maxval = matrix.Max()
        pixels = img.load()

        # Iterating just over the non-zero data points is ideal when
        # plotting the whole image, but for generating tile sets, it might
        # make more sense for the caller to partition the points and pass in
        # a list of points to use for each image.  That way we only iterate
        # over the points once, rather than once per image.  That also gives
        # the caller an opportunity to do something better for tiles that
        # contain no data.
        for ((x, y), val) in matrix.iteritems():
            if bounding_box.IsInside((x, y)):
                if self.background:
                    pixels[x - minX, y - minY] = _blend_pixels(
                        self.colormap[int(255 * val / maxval)],
                        self.background)
                else:
                    pixels[x - minX, y - minY] = self.colormap[int(
                        255 * val / maxval)]
        if self.background_image:
            # Is this really the best way?
            img = Image.composite(img, self.background_image, img.split()[3])
        img.save(filename)


class ImageSeriesMaker():
    def __init__(
            self, colormap, background, background_image, filename_template,
            num_frames, total_points, width, height, bounding_box):
        self.image_maker = ImageMaker(colormap, background, background_image)
        self.filename_template = filename_template
        self.num_frames = num_frames
        self.frequency = float(num_frames) / total_points
        self.input_count = 0
        self.frame_count = 0
        self.width = width
        self.height = height
        self.bounding_box = bounding_box

    def MaybeSaveImage(self, matrix):
        self.input_count += 1
        x = self.input_count * self.frequency   # frequency <= 1
        if x - int(x) < self.frequency:
            self.frame_count += 1
            logging.info(
                'Frame %d of %d' % (self.frame_count, self.num_frames))
            matrix = matrix.Finalized()
            self.image_maker.SavePNG(
                matrix, self.filename_template % self.frame_count,
                self.width, self.height, self.bounding_box)


def _GetOSMImage(bbox, zoom):
    # Just a wrapper for osm.createOSMImage to translate coordinate schemes
    try:
        from osmviz.manager import PILImageManager, OSMManager
        osm = OSMManager(
            image_manager=PILImageManager('RGB'),
            server=options.osm_base)
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
    proj.SetScale(scale)
    logging.info('Scale: %f' % (111319.5 / scale))
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


def GetOSMBackground(bbox_ll, padding):
    zoom = ChooseOSMZoom(bbox_ll, padding)
    proj = MercatorProjection()
    proj.SetScale(_ScaleForOSMZoom(zoom))
    bbox_xy = bbox_ll.Map(proj.Project)
    # We're not checking that the padding fits within the specified size.
    bbox_xy.Grow(padding)
    bbox_ll = bbox_xy.Map(proj.InverseProject)
    image, img_bbox_ll = _GetOSMImage(bbox_ll, zoom)
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


def _8bitInt_to_float(i):
    '''Primirily for scaling numbers from [0,255] to [0,1.0].  It will
    also work on numbers outside that range, but with some skew: every
    256th place is ignored so that people writing in hex can write 1XX
    in order to get 1.0 + _8bitInt_to_float(XX).  This is mathematically
    incorrect, but rather convenient.  The only time anyone will give a
    number outside [0,255] is on the command line, using strings like
    #120ffffff, where a non-zero first digit lets hue wrap around the
    color wheel the opposite way.  ff would otherwise be equivalent to
    1fe, not 1ff.'''
    return float(i - int(i / 256)) / 255


def ProcessShapes(shapes, projection, hook=None):
    matrix = Matrix.MatrixFactory(options.decay)
    logging.info('processing data')
    kernel = kernels[options.kernel](options.radius)
    for shape in shapes:
        shape = shape.Map(projection.Project)
        shape.AddHeatToMatrix(matrix, kernel)
        if hook:
            hook(matrix)
    return matrix


def setup_options():
    # handy for other programs that use this as a module
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
            '(for square mercator: -85.0511,-180,85.0511,180)'))
    optparser.add_option(
        '-R', '--margin', metavar='INT', type='int', default=0,
        help=(
            'Try to keep data at least this many pixels away from image'
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
        '-f', '--frames', type='int', default=30,
        help='number of frames for animation; default: %default')
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
    return optparser

# Note to self: -m #0aa80ff00 -M #120ffffff is nice.


def main():
    global options

    logging.basicConfig(format='%(relativeCreated)8d ms  // %(message)s')
    optparser = setup_options()
    (options, args) = optparser.parse_args()

    if options.verbose:
        logging.getLogger().setLevel(logging.INFO)

    if not ((options.points or options.gpx or options.csv or options.load)
            and (options.output or options.save)):
        sys.stderr.write(
            "You must specify one input (-g -p --csv -L) and at least one "
            "output (-o or -S).\n")
        sys.exit(1)

    if ((options.gpx or options.points or options.csv)
        and not ((options.width or options.height or options.scale
                  or options.background_image)
                 or (options.osm and options.zoom))):
        sys.stderr.write(
            "With --gpx, --points or --csv, you must also specify at least "
            "one of --width, --height,\n --scale, or --background_image, or "
            "both --osm and --zoom.\n")
        sys.exit(1)

    if options.output:
        colormap = ColorMap()
        if options.gradient:
            colormap.FromImage(Image.open(options.gradient))
        else:
            colormap.FromHsvaRangeStrings(options.hsva_min, options.hsva_max)

    matrix = None  # make the result available for load & save
    if options.load:
        logging.info('loading data')
        process_data = False
        import cPickle as pickle
        matrix = pickle.load(open(options.load))
    else:
        process_data = True
        if options.gpx:
            track = TrackLog(options.gpx)
            shapes = []
            for trkseg in track.segments:
                for i, p1 in enumerate(trkseg[:-1]):
                    p2 = trkseg[i + 1]
                    # We'll end up projecting every point twice, but this is
                    # the least of our performance problems.
                    shapes.append(LineSegment(p1.coords, p2.coords))
        elif options.points:
            logging.info('reading points')
            shapes = []
            with open(options.points, 'rU') as f:
                for line in f:
                    line = line.strip()
                    if len(line) > 0:  # ignore blank lines
                        values = [float(x) for x in line.split()]
                        assert len(values) == 2 or len(values) == 3, (
                            'input lines must have two or three values: %s' % line)
                        (lat, lon) = values[0:2]
                        weight = 1.0 if len(values) == 2 else values[2]
                        shapes.append(Point((lat, lon), weight))
                logging.info('read %d points' % len(shapes))
        else:
            logging.info('reading csv')
            import csv
            shapes = []
            with open(options.csv, 'rU') as f:
                reader = csv.reader(f)
                if options.ignore_csv_header:
                    reader.next()  # Skip header line
                for row in reader:
                    (lat, lon) = (float(row[0]), float(row[1]))
                    shapes.append(Point((lat, lon)))
                logging.info('read %d points' % len(shapes))

    logging.info('Determining scale and scope')

    bounding_box_ll = None
    bounding_box_xy_padding = options.margin
    projection = None
    if options.extent:
        bounding_box_ll = BoundingBox(string=options.extent)
        # TODO: (speed optimization) we should compute a bounding box that
        # includes an extra kernel radius and use it to discard points that
        # are too far outside the extent to affect the output.
    elif options.load:
        projection = matrix.projection
        bounding_box_ll = matrix.BoundingBox().Map(projection.InverseProject)
    else:
        bounding_box_ll = BoundingBox(shapes=shapes)
        bounding_box_xy_padding += options.radius   # Make room for the spread

    background_image = None
    if options.background_image:
        background_image = Image.open(options.background_image)
        (options.width, options.height) = background_image.size
    elif options.osm:
        background_image, bounding_box_ll, projection = GetOSMBackground(
            bounding_box_ll, bounding_box_xy_padding)
        (options.width, options.height) = background_image.size
        bounding_box_xy_padding = 0  # already baked in

    if options.background_brightness:
        background_image = background_image.point(
            lambda x: x * options.background_brightness)

    if not projection:
        projection = projections[options.projection]()
        projection.AutoSetScale(bounding_box_ll, bounding_box_xy_padding)
    bounding_box_xy = bounding_box_ll.Map(projection.Project)
    bounding_box_xy.Grow(bounding_box_xy_padding)
    if not options.extent:
        logging.info('Map extent: %s' % bounding_box_xy.Map(
            projection.InverseProject).Extent())

    if process_data:
        if options.animate:
            import tempfile
            import os.path
            import shutil
            import subprocess
            tmpdir = tempfile.mkdtemp()
            logging.info('Putting animation frames in %s' % tmpdir)
            imgfile_template = os.path.join(tmpdir, 'frame-%05d.png')
            maker = ImageSeriesMaker(
                colormap, options.background, background_image,
                imgfile_template, min(options.frames, len(shapes)),
                len(shapes), options.width, options.height, bounding_box_xy)
            hook = maker.MaybeSaveImage
            matrix = ProcessShapes(shapes, projection, hook)
            if maker.frame_count < options.frames:
                hook(matrix)  # one last one
            command = ['ffmpeg', '-i', imgfile_template]
            if options.ffmpegopts:
                # I hope they don't have spaces in their arguments
                command.extend(options.ffmpegopts.split())
            # output filename must be last
            command.append(options.output)
            logging.info('Encoding video: %s' % ' '.join(command))
            subprocess.call(command)
            if not options.keepframes:
                shutil.rmtree(tmpdir)
            else:
                logging.info('The animation frames are in %s' % tmpdir)
        else:
            matrix = ProcessShapes(shapes, projection)
            matrix = matrix.Finalized()
    if options.output and not options.animate:
        ImageMaker(colormap, options.background, background_image).SavePNG(
            matrix, options.output, options.width, options.height,
            bounding_box_xy)

    if options.save:
        logging.info('saving data')
        import cPickle as pickle
        matrix.projection = projection
        pickle.dump(matrix, open(options.save, 'w'), 2)

    logging.info('end')

if __name__ == '__main__':
    main()
