#!/bin/sh

coverage run --source=. -m pytest \
  && coverage report
pycodestyle *.py test/*.py
pyflakes *.py test/*.py
