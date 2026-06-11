install:
	pip install -e ".[dev]"

# ── Build ─────────────────────────────────────────────────────────────────────
clean:
	rm -rf dist/ build/
	find . -name "__pycache__" -type d -exec rm -rf {} +
	find . -name "*.pyc" -delete

build: clean
	python -m build
	python -m twine check --strict dist/*

# ── Tests ─────────────────────────────────────────────────────────────────────
test:
	pytest tests/test_tversky_lib.py tests/test_v2_equivalence.py -v

test-mnist:
	pytest tests/test_mnist.py -v
