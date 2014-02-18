all: tests python3_tests

tests:
	test/test_animation.py
	test/test.py
	test/test_gradients.py
	test/test_projection_scale.py

python3_tests:
	python3 test/test_animation.py
	python3 test/test.py
	python3 test/test_gradients.py
	python3	test/test_projection_scale.py
