.PHONY: test type eval serve
test:
	python -m pytest -q
type:
	python -m mypy kagent
eval:
	python -m kagent.evalrun
serve:
	python -m kagent.server
