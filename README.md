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

There's another kind of heatmap, also called a cloropleth map, in
which you divide your map into regions and color each region to
indicate something.  This tool is not for that.

A more thorough description and examples are posted at
 <http://sethoscope.net/heatmap/>


# What next?

Improvements I might one day get to, probably in this order:

 - refactor into a useful library (rather than a command line tool)
 - weight GPS line segments by inverse speed (so each point properly
   represents how long was spent there, regardless of data sampling
   rate)
 - pace animation according to data timestamps, rather than uniformly
 - provide some decent way to put results on a slippy map

