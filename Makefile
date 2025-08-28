PY := python

.PHONY: eval tests

eval:
	$(PY) eval/runner.py --out eval/report.md | cat

tests:
	pytest -q


