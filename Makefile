##
# Variables
##

ENV_NAME = env
ENV_ACT = . env/bin/activate;
PIP = $(ENV_NAME)/bin/pip
PY = $(ENV_NAME)/bin/python
PYTEST_ARGS = --doctest-modules -v -s --hypothesis-profile dev
PYTEST_TARGET = catenary tests
COVERAGE_ARGS = --cov-config setup.cfg --cov-report term-missing --cov
COVERAGE_TARGET = catenary


##
# Targets
##

.PHONY: build
build: clean install

.PHONY: clean
clean: clean-env clean-files

.PHONY: clean-env
clean-env:
	rm -rf $(ENV_NAME)

.PHONY: clean-files
clean-files:
	rm -rf .tox
	rm -rf .coverage
	find . -name \*.pyc -type f -delete
	find . -name \*.test.db -type f -delete
	find . -depth -name __pycache__ -type d -exec rm -rf {} \;
	rm -rf dist *.egg* build

.PHONY: install
install:
	rm -rf $(ENV_NAME)
	virtualenv -p python3 $(ENV_NAME)
	$(PIP) install -r requirements-dev.txt -r requirements.txt && $(PIP) install -e .[cpu]

.PHONY: test
test: lint pytest

.PHONY: pytest
pytest:
	$(ENV_ACT) pytest $(PYTEST_ARGS) $(COVERAGE_ARGS) $(COVERAGE_TARGET) $(PYTEST_TARGET)

.PHONY: repl
repl:
	$(ENV_ACT) ipython

.PHONY: lint
lint: pre-commit

.PHONY: pre-commit
pre-commit: $(ENV_ACT) pre-commit run --all-files

.PHONY: push
push:
	git push origin master
	git push --tags

.PHONY: notebook
notebook:
	$(ENV_ACT) jupyter lab
