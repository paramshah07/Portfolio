run_tests:
	python -m unittest discover .
	coverage run --source=common,ai_algorithm,finance_algorithm -m unittest discover -s .
	coverage report -m