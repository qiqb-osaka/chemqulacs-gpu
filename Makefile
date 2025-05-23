PYTEST := poetry run pytest
FORMATTER := poetry run ruff format
LINTER := poetry run ruff check
TYPE_CHECKER := poetry run mypy
SPHINX_APIDOC := poetry run sphinx-apidoc

PROJECT_DIR := chemqulacs
CHECK_DIR := $(PROJECT_DIR) tests
PORT := 8000

# If this project is not ready to pass mypy, remove `type` below.
.PHONY: check
check: format lint type

.PHONY: ci
ci: format_check lint_check type

.PHONY: test
test:
	$(PYTEST) -v

tests/%.py: FORCE
	$(PYTEST) $@

# Idiom found at https://www.gnu.org/software/make/manual/html_node/Force-Targets.html
FORCE:

.PHONY: format
format:
	$(FORMATTER) $(CHECK_DIR)

.PHONY: format_check
format_check:
	$(FORMATTER) $(CHECK_DIR) --check --diff

.PHONY: lint
lint:
	$(LINTER) $(CHECK_DIR) --fix

.PHONY: lint_check
lint_check:
	$(LINTER) $(CHECK_DIR)

.PHONY: type
type:
	$(TYPE_CHECKER)  @mypy_files.txt

.PHONY: serve
serve: html
	poetry run python -m http.server --directory doc/build/html $(PORT)

.PHONY: doc
html: api
	poetry run $(MAKE) -C doc html

.PHONY: api
api:
	$(SPHINX_APIDOC) -f -e -o doc/source $(PROJECT_DIR)
