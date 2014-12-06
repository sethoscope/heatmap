#!/bin/bash
# install gdal

sudo add-apt-repository -y ppa:ubuntugis/ubuntugis-unstable
sudo apt-get update -qq
sudo apt-get install libgdal1h
sudo apt-get install -y python-gdal python3-gdal
