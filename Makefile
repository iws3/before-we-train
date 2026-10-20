PY ?= python3

.PHONY: setup data ch1 test clean

setup:            ## install dependencies
	$(PY) -m pip install -r requirements.txt

data:             ## download the public-domain text used in 1.2.4 and 1.3.5
	$(PY) data/get_data.py

ch1:              ## run every chapter-01 script in order
	@for f in chapter-01/1_*.py; do echo "\n===== $$f ====="; $(PY) $$f || exit 1; done

test:             ## run the test-suite
	$(PY) -m pytest -q

clean:
	rm -rf chapter-*/outputs/* .pytest_cache __pycache__ */__pycache__
	touch chapter-01/outputs/.gitkeep
