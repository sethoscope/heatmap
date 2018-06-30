# heatmap

## why?

There are a few kinds of heat maps. This program is for when you have
data points, each with a pair of orthogonal coordinates (X/Y, lat-lon)
and you want to plot them on a map such that they blob together a bit
to indicate density.

So, it's good for things like:

 - eye tracking data
 - lat/lon geocoded data points
 - GPS tracks

## why not?

It's not good for:

 - showing results in realtime (because it's too slow)
 - running in a browser (because it's in Python)
 - automatically layering on proprietary map systems

So... why use a slow data visualizer that doesn't run in a browser?
Because the output looks better.

There's another kind of heatmap, also called a choropleth map, in
which you divide your map into regions and color each region to
indicate something.  This tool is not for that.

A more thorough description and examples are posted at
 <http://sethoscope.net/heatmap/>

## change to command line

As of version 1.13 (April 2017), the method of specifying input files
has changed. The old way will continue to be supported for at least
one year, but the new way is nicer.
Read [this tutorial](CMDLINE_UPDATING.md) for examples and help on
switching to the new way.

## Tests

[![Build Status](https://travis-ci.org/sethoscope/heatmap.png?branch=master)](https://travis-ci.org/sethoscope/heatmap)
[![Coverage Status](https://coveralls.io/repos/sethoscope/heatmap/badge.png?branch=master)](https://coveralls.io/r/sethoscope/heatmap?branch=master)

## installation

On Debian Linux, required packages can be installed by:
`apt install python-pil:amd64 python-wheel python-setuptools python-pip; pip install osmviz`
