# specifying input files

Before version 1.13 (released in April 2017), you could only provide
one input file, and the flag used to provide it depended on its type
(CSV, GPX, ESRI Shapefile, etc).  Commands looked like this:

    heatmap.py --points plainfile -o map.png --width 100
    heatmap.py --csv points.csv -o map.png --width 100
    heatmap.py --gpx track.gpx -o map.png --width 100
    heatmap.py --shp_file points.shp -o map.png --width 100

As of version 1.13, the equivalent of those four commands would look like this:

    heatmap.py -o map.png --width 100 plainfile
    heatmap.py -o map.png --width 100 points.csv
    heatmap.py -o map.png --width 100 track.gpx
    heatmap.py -o map.png --width 100 points.shp

Moreover, you can now specify multiple input files, and they need not
be the same type.

    heatmap.py -o map.png --width 100 track*.gpx extras.csv morepoints.shp

The type of the input file is guessed from the filename extension
(defaulting to the original plain type), but if your files are
named oddly and you need to override the type, you can do that with
`--filetype`.

    heatmap.py -o map.png --width 100 --filetype gpx *.xml

