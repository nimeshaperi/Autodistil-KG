#!/usr/bin/env bash
# =============================================================================
# setup-hetionet.sh — Import the Hetionet biomedical knowledge graph into Neo4j 5
# =============================================================================
#
# WHAT IT DOES
#   1. Starts the legacy dhimmel/hetionet container (Neo4j 3.5) on a temporary
#      port to serve as the data source.
#   2. Exports every node label and relationship type to CSV files using the
#      Neo4j HTTP Transactional API (no APOC or extra driver needed).
#   3. Imports those CSVs into the Neo4j 5 data volume used by Autodistil-KG
#      via `neo4j-admin database import full`.
#   4. Cleans up the temporary legacy container and CSV files.
#
# PREREQUISITES
#   - Docker installed and running
#   - The autodistil-kg-neo4j container must be stopped (we write to its volume)
#     OR not yet started (first-time setup):
#       docker compose stop neo4j      # if already running
#   - Python 3 available on the host (used for the CSV export step)
#   - Internet access to pull dhimmel/hetionet (~1 GB) on first run
#
# USAGE
#   bash scripts/setup-hetionet.sh
#
# EXPECTED RESULT
#   "IMPORT DONE ... Imported: 47031 nodes, 2250197 relationships"
#   After this, run:  docker compose up -d
#
# =============================================================================

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# ── Configuration ─────────────────────────────────────────────────────────────
LEGACY_IMAGE="dhimmel/hetionet"
LEGACY_CONTAINER="hetionet-legacy-export"
LEGACY_HTTP_PORT=7476          # Temporary HTTP port for the legacy Neo4j 3.5
LEGACY_BOLT_PORT=7478          # Temporary Bolt port

NEO4J5_VOLUME="autodistil-kg_neo4j_data"   # Must match the volume name in docker-compose.yml
NEO4J5_IMAGE="neo4j:5"
NEO4J5_CONTAINER="autodistil-kg-neo4j"
NEO4J5_PASSWORD="${NEO4J_PASSWORD:-password}"

EXPORT_DIR="$(mktemp -d)"
trap 'echo "Cleaning up…"; rm -rf "$EXPORT_DIR"; docker rm -f "$LEGACY_CONTAINER" 2>/dev/null || true' EXIT

# ── Helpers ───────────────────────────────────────────────────────────────────
log()  { echo "  [hetionet] $*"; }
die()  { echo "ERROR: $*" >&2; exit 1; }
need() { command -v "$1" &>/dev/null || die "$1 is required but not found in PATH"; }

need docker
need python3

# ── Step 0: Pre-flight checks ─────────────────────────────────────────────────
log "Checking prerequisites…"

# Make sure the main Neo4j 5 container is not running (we're about to write its volume)
if docker ps --format '{{.Names}}' | grep -q "^${NEO4J5_CONTAINER}$"; then
    die "The ${NEO4J5_CONTAINER} container is currently running.\n       Stop it first:  docker compose stop neo4j\n       Then re-run this script."
fi

# Check if data already looks imported (quick count via a one-shot container)
log "Checking if Hetionet data is already imported…"
NODE_COUNT=$(docker run --rm \
    -v "${NEO4J5_VOLUME}:/data" \
    --entrypoint "" \
    "${NEO4J5_IMAGE}" \
    bash -c "cypher-shell --non-interactive -u neo4j -p '${NEO4J5_PASSWORD}' \
             -a bolt://localhost:7687 'MATCH (n) RETURN count(n) AS c' 2>/dev/null | tail -1" \
    2>/dev/null || echo "0")

if [[ "$NODE_COUNT" =~ ^[0-9]+$ ]] && [ "$NODE_COUNT" -gt 40000 ]; then
    log "Database already has ${NODE_COUNT} nodes — skipping import."
    log "If you want to re-import, delete the volume first:"
    log "  docker volume rm ${NEO4J5_VOLUME}"
    exit 0
fi

# ── Step 1: Start the legacy Hetionet container ───────────────────────────────
log "Pulling ${LEGACY_IMAGE} (this may take a few minutes on first run)…"
docker pull "${LEGACY_IMAGE}"

log "Starting legacy Hetionet container on ports ${LEGACY_HTTP_PORT}/${LEGACY_BOLT_PORT}…"
docker rm -f "${LEGACY_CONTAINER}" 2>/dev/null || true
docker run -d \
    --name "${LEGACY_CONTAINER}" \
    -p "${LEGACY_HTTP_PORT}:7474" \
    -p "${LEGACY_BOLT_PORT}:7687" \
    -e NEO4J_AUTH=none \
    "${LEGACY_IMAGE}"

# Wait until Neo4j 3.5 is accepting connections
log "Waiting for legacy Neo4j to be ready…"
for i in $(seq 1 60); do
    if curl -sf "http://localhost:${LEGACY_HTTP_PORT}/db/data/" >/dev/null 2>&1; then
        log "Legacy Neo4j is ready (attempt ${i})."
        break
    fi
    sleep 3
    if [ "$i" -eq 60 ]; then
        die "Legacy Neo4j did not become ready after 3 minutes."
    fi
done

# ── Step 2: Export all nodes and relationships to CSV ─────────────────────────
log "Exporting Hetionet data to CSV files in ${EXPORT_DIR}…"

python3 - <<'PYEOF' "$EXPORT_DIR" "$LEGACY_HTTP_PORT"
"""
Export all nodes (by label) and relationships (by type) from a Neo4j 3.x
HTTP API to CSV files compatible with neo4j-admin database import.

Node CSV format:    :ID(neo4j-id), <props…>, :LABEL
Rel   CSV format:   :START_ID(neo4j-id), :END_ID(neo4j-id), :TYPE, <props…>
"""
import sys, csv, json, time, re
import urllib.request, urllib.error

EXPORT_DIR  = sys.argv[1]
HTTP_PORT   = sys.argv[2]
BASE_URL    = f"http://localhost:{HTTP_PORT}"
BATCH_SIZE  = 5000   # rows per Cypher request


def cypher(query, params=None):
    payload = json.dumps({
        "statements": [{"statement": query,
                         "parameters": params or {},
                         "resultDataContents": ["row"]}]
    }).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/db/data/transaction/commit",
        data=payload,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
    if data.get("errors"):
        raise RuntimeError(data["errors"])
    return data["results"][0]


def safe_name(s):
    """Strip non-alphanumeric chars for use as a CSV filename."""
    return re.sub(r"[^A-Za-z0-9_]", "_", s)


print("  Querying node labels…")
labels_result = cypher("MATCH (n) RETURN DISTINCT labels(n)[0] AS lbl ORDER BY lbl")
labels = [row[0] for row in labels_result["data"] if row[0]]

for label in labels:
    print(f"  Exporting nodes: {label}…")

    # Discover all property keys for this label
    keys_result = cypher(f"MATCH (n:{label}) WITH keys(n) AS k UNWIND k AS key RETURN DISTINCT key ORDER BY key")
    prop_keys = [row[0] for row in keys_result["data"]]

    fname = f"{EXPORT_DIR}/nodes_{safe_name(label)}.csv"
    with open(fname, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        # Header row: :ID, then each property, then :LABEL
        writer.writerow([":ID(neo4j-id)"] + prop_keys + [":LABEL"])

        skip = 0
        while True:
            q = (f"MATCH (n:{label}) "
                 f"RETURN id(n), {', '.join(f'n.`{k}`' for k in prop_keys) if prop_keys else 'null'} "
                 f"ORDER BY id(n) SKIP {skip} LIMIT {BATCH_SIZE}")
            result = cypher(q)
            rows = result["data"]
            if not rows:
                break
            for row in rows:
                node_id = row[0]
                props   = row[1:] if prop_keys else []
                writer.writerow([node_id] + list(props) + [label])
            if len(rows) < BATCH_SIZE:
                break
            skip += BATCH_SIZE

print("  Querying relationship types…")
rel_types_result = cypher("MATCH ()-[r]->() RETURN DISTINCT type(r) AS t ORDER BY t")
rel_types = [row[0] for row in rel_types_result["data"] if row[0]]

for rel_type in rel_types:
    print(f"  Exporting relationships: {rel_type}…")

    keys_result = cypher(f"MATCH ()-[r:{rel_type}]->() WITH keys(r) AS k UNWIND k AS key RETURN DISTINCT key ORDER BY key LIMIT 1")
    prop_keys = [row[0] for row in keys_result["data"]]

    fname = f"{EXPORT_DIR}/rels_{safe_name(rel_type)}.csv"
    with open(fname, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow([":START_ID(neo4j-id)", ":END_ID(neo4j-id)", ":TYPE"] + prop_keys)

        skip = 0
        while True:
            q = (f"MATCH (a)-[r:{rel_type}]->(b) "
                 f"RETURN id(a), id(b), {', '.join(f'r.`{k}`' for k in prop_keys) if prop_keys else 'null as _dummy'} "
                 f"ORDER BY id(a) SKIP {skip} LIMIT {BATCH_SIZE}")
            result = cypher(q)
            rows = result["data"]
            if not rows:
                break
            for row in rows:
                start_id, end_id = row[0], row[1]
                props = row[2:] if prop_keys else []
                writer.writerow([start_id, end_id, rel_type] + list(props))
            if len(rows) < BATCH_SIZE:
                break
            skip += BATCH_SIZE

print(f"  Export complete. Files written to {EXPORT_DIR}")
PYEOF

log "Export complete. CSV files in ${EXPORT_DIR}:"
ls -lh "${EXPORT_DIR}"

# ── Step 3: Stop the legacy container ────────────────────────────────────────
log "Stopping legacy Hetionet container…"
docker stop "${LEGACY_CONTAINER}" >/dev/null

# ── Step 4: Clear any stale Neo4j 5 database files ───────────────────────────
log "Clearing stale data from the Neo4j 5 volume (if any)…"
docker run --rm \
    -v "${NEO4J5_VOLUME}:/data" \
    "${NEO4J5_IMAGE}" \
    bash -c "rm -rf /data/databases/neo4j /data/transactions/neo4j 2>/dev/null; true"

# ── Step 5: Build and run neo4j-admin import ─────────────────────────────────
log "Building neo4j-admin import command…"

IMPORT_CMD="neo4j-admin database import full \
    --overwrite-destination \
    --multiline-fields=true \
    --skip-bad-relationships=true \
    --skip-duplicate-nodes=true \
    --trim-strings=true"

for f in "${EXPORT_DIR}"/nodes_*.csv; do
    label=$(basename "$f" | sed 's/nodes_//;s/\.csv//')
    IMPORT_CMD="${IMPORT_CMD} --nodes=${label}=/import/$(basename "$f")"
done

for f in "${EXPORT_DIR}"/rels_*.csv; do
    IMPORT_CMD="${IMPORT_CMD} --relationships=/import/$(basename "$f")"
done

IMPORT_CMD="${IMPORT_CMD} -- neo4j"

log "Running neo4j-admin import (this may take 1–3 minutes)…"
docker run --rm \
    -v "${EXPORT_DIR}:/import:ro" \
    -v "${NEO4J5_VOLUME}:/data" \
    "${NEO4J5_IMAGE}" \
    bash -c "${IMPORT_CMD}"

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "======================================================================"
echo "  Hetionet import complete!"
echo ""
echo "  Expected: Imported: 47031 nodes, 2250197 relationships"
echo ""
echo "  Next steps:"
echo "    docker compose up -d          # start the full stack"
echo "    open http://localhost:3000    # Autodistil-KG UI"
echo "    open http://localhost:7474    # Neo4j Browser"
echo "======================================================================"
