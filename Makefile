all: tests python3_tests

tests:
	test/test.py
	test/test_animation.py
	test/test_configuration.py
	test/test_coordinates.py
	test/test_gradients.py
	test/test_projection_scale.py
	test/test_projections.py
	test/test_random.py
	test/test_system.py
	test/test_system_gpx.py

python3_tests:
	python3 test/test_animation.py
	python3 test/test_coordinates.py
	python3 test/test_gpx.py
	python3 test/test_gradients.py
	python3	test/test_projection_scale.py
	python3 test/test_projections.py
	python3 test/test.py
	python3 test/test_random.py
