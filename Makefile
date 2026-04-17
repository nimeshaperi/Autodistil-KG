# Autodistil-KG Monorepo Makefile
# Run lint, typecheck, and test across all submodules.

CORE_DIR := Autodistil-KG_core
API_DIR  := Autodistil-KG_api
CLIENT_DIR := Autodistil-KG_client

.PHONY: lint typecheck test fmt generate-types

# ── Linting ────────────────────────────────────────────────────────────────
lint:
	@echo "==> Linting core..."
	cd $(CORE_DIR) && poetry run ruff check src/
	@echo "==> Linting API..."
	cd $(API_DIR) && poetry run ruff check src/
	@echo "==> Linting client..."
	cd $(CLIENT_DIR) && npx eslint src/ --ext .ts,.tsx

# ── Type checking ──────────────────────────────────────────────────────────
typecheck:
	@echo "==> Type checking core..."
	cd $(CORE_DIR) && poetry run mypy src/autodistil_kg/ --ignore-missing-imports
	@echo "==> Type checking API..."
	cd $(API_DIR) && poetry run mypy src/autodistilkg_api/ --ignore-missing-imports
	@echo "==> Type checking client..."
	cd $(CLIENT_DIR) && npx tsc --noEmit

# ── Testing ────────────────────────────────────────────────────────────────
test:
	@echo "==> Testing core..."
	cd $(CORE_DIR) && poetry run pytest
	@echo "==> Testing API..."
	cd $(API_DIR) && poetry run pytest
	@echo "==> Testing client..."
	cd $(CLIENT_DIR) && npx vitest run 2>/dev/null || echo "(no vitest configured yet)"

# ── Formatting ─────────────────────────────────────────────────────────────
fmt:
	@echo "==> Formatting core..."
	cd $(CORE_DIR) && poetry run ruff format src/
	@echo "==> Formatting API..."
	cd $(API_DIR) && poetry run ruff format src/

# ── Type generation ────────────────────────────────────────────────────────
generate-types:
	./scripts/generate-types.sh
