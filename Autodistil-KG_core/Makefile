.PHONY: install run run-default run-help help

help:
	@echo "Autodistil-KG - Build instruction datasets from knowledge graphs"
	@echo ""
	@echo "Usage:"
	@echo "  make install      Install dependencies (run once)"
	@echo "  make run          Run with default config (no args needed)"
	@echo "  make run-default  Same as 'make run'"
	@echo "  make run-help     Show CLI help"
	@echo ""
	@echo "Or use poetry directly:"
	@echo "  poetry run python -m autodistil_kg.run"
	@echo "  poetry run autodistil-kg --help"

install:
	poetry install

run run-default:
	poetry run python run.py

run-help:
	poetry run python run.py --help
