#!/bin/sh

uv run coverage run --source=. -m pytest \
  && coverage report
pycodestyle src/heatmap77/heatmap.py test/*.py examples/*.py
pyflakes src/heatmap77/heatmap.py test/*.py examples/*.py
