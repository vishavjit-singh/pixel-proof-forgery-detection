PY ?= python3
export PYTHONPATH := src

.PHONY: install demo generate train evaluate report test lint serve clean

install:
	$(PY) -m pip install -r requirements-dev.txt

## Synthetic benchmark → NumPy patch CNN + classifiers → test-split evaluation → README charts
demo: generate train evaluate report

generate:
	$(PY) -m pixelproof generate

train:
	$(PY) -m pixelproof train

evaluate:
	$(PY) -m pixelproof evaluate > /dev/null && cat data/eval/metrics.json | head -40

report:
	$(PY) -m pixelproof report

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check .

serve:
	$(PY) -m pixelproof serve

clean:
	rm -rf data models
