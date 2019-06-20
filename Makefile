all: test

lint: pylint pydoc pycode

pylint:
	pylint --rcfile=.pylintrc src/hnccorr

pydoc:
	pydocstyle src/

pycode:
	pycodestyle src/ tests/

test:
	pytest

integration:
	pytest tests/integration

tags:
	ctags -R language-force=python src
