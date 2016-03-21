# This simple Makefile gives me an easy way to run tests locally.
# It's not maintained as consistently as .travis.yml and doesn't
# run as many tests (e.g. code coverage), but is easy to use without
# having to commit and push code to where Travis can see it.
#
# It's only intended for my own use, but I've included it in the repo
# anyway in case someone else finds it helpful.

tests: python2_tests python3_tests

python2_tests:
	python2 test/test.py
	python2 test/test_animation.py
	python2 test/test_configuration.py
	python2 test/test_coordinates.py
	python2 test/test_gradients.py
	python2 test/test_projection_scale.py
	python2 test/test_projections.py
	python2 test/test_random.py
	python2 test/test_system.py
	python2 test/test_system_gpx.py
	pep8 heatmap.py test/*.py
	pyflakes heatmap.py test/*.py

python3_tests:
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
