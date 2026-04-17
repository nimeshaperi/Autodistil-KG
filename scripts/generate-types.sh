#!/usr/bin/env bash
# Generate TypeScript types from the FastAPI OpenAPI spec.
#
# Usage:  ./scripts/generate-types.sh
#
# Prerequisites:
#   - The API must be installable (poetry install in Autodistil-KG_api)
#   - Node.js must be available (for npx openapi-typescript)
#
# Outputs:
#   Autodistil-KG_client/src/types/generated-api.ts   -- API request/response types
#   Autodistil-KG_client/src/types/generated-events.ts -- Pipeline event types

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
API_DIR="$ROOT/Autodistil-KG_api"
CORE_DIR="$ROOT/Autodistil-KG_core"
CLIENT_TYPES_DIR="$ROOT/Autodistil-KG_client/src/types"

echo "==> Exporting OpenAPI spec from FastAPI..."
OPENAPI_JSON="$ROOT/scripts/openapi.json"
cd "$API_DIR"
poetry run python -c "
from autodistilkg_api.main import app
import json, sys
json.dump(app.openapi(), sys.stdout, indent=2)
" > "$OPENAPI_JSON"

echo "==> Generating TypeScript types from OpenAPI spec..."
cd "$ROOT/Autodistil-KG_client"
npx openapi-typescript "$OPENAPI_JSON" -o "$CLIENT_TYPES_DIR/generated-api.ts"

echo "==> Exporting event schemas from core package..."
EVENT_SCHEMAS_JSON="$ROOT/scripts/event-schemas.json"
cd "$CORE_DIR"
poetry run python -c "
from autodistil_kg.pipeline.events import get_event_schemas
import json, sys
json.dump(get_event_schemas(), sys.stdout, indent=2)
" > "$EVENT_SCHEMAS_JSON"

echo "==> Generating TypeScript types from event schemas..."
cd "$ROOT/Autodistil-KG_client"
npx json-schema-to-typescript "$EVENT_SCHEMAS_JSON" -o "$CLIENT_TYPES_DIR/generated-events.ts" 2>/dev/null || {
    echo "    (json-schema-to-typescript not found -- install with: npm i -g json-schema-to-typescript)"
    echo "    Event schemas written to $EVENT_SCHEMAS_JSON for manual conversion."
}

echo "==> Done. Generated types written to:"
echo "    $CLIENT_TYPES_DIR/generated-api.ts"
echo "    $CLIENT_TYPES_DIR/generated-events.ts (if json-schema-to-typescript was available)"
