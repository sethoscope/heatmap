# This simple Makefile gives me an easy way to run tests locally.
# It's not maintained as consistently as .travis.yml and doesn't
# run as many tests (e.g. code coverage), but is easy to use without
# having to commit and push code to where Travis can see it.
#
# It's only intended for my own use, but I've included it in the repo
# anyway in case someone else finds it helpful.

all: tests coverage style

test: tests
tests:
	python3 test/test.py
	python3 test/test_animation.py
	python3 test/test_configuration.py
	python3 test/test_coordinates.py
	python3 test/test_gradients.py
	python3 test/test_projection_scale.py
	python3 test/test_projections.py
	python3 test/test_random.py
	python3 test/test_system.py
	python3 test/test_system_gpx.py
	python3 test/test_system_csv.py
	python3 test/test_shp_file.py
	python3 test/test_old_cmdline_support.py

coverage_report:
	python3 -m coverage report

coverage: tests coverage_report

style:
	pycodestyle *.py test/*.py
