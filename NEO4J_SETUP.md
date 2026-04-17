# Neo4j 5 Setup — Hetionet Dataset

This guide covers setting up the Hetionet biomedical knowledge graph on Neo4j 5 for use with Autodistil-KG.

## Prerequisites

- Docker installed and running
- Ports 7475 (HTTP) and 7688 (Bolt) available

## 1. Pull Neo4j 5 Image

```bash
docker pull neo4j:5
```

## 2. Import Hetionet Data

The exported CSV files are located in `/tmp/hetionet-export/`. If you need to re-export from the original Hetionet container, see [Re-exporting from Hetionet](#re-exporting-from-hetionet) below.

### Create the data volume and run the import

```bash
EXPORT_DIR="/tmp/hetionet-export"
VOLUME_NAME="autodistil-kg_neo4j_data"

# Clear any previous data
docker run --rm \
  -v "$VOLUME_NAME:/data" \
  neo4j:5 bash -c "rm -rf /data/databases/neo4j /data/transactions/neo4j"

# Build and run the import command
CMD="neo4j-admin database import full --overwrite-destination --multiline-fields=true --skip-bad-relationships=true --skip-duplicate-nodes=true --trim-strings=true"

for f in "$EXPORT_DIR"/nodes_*.csv; do
    label=$(basename "$f" | sed 's/nodes_//;s/\.csv//')
    CMD="$CMD --nodes=$label=/import/$(basename "$f")"
done

for f in "$EXPORT_DIR"/rels_*.csv; do
    CMD="$CMD --relationships=/import/$(basename "$f")"
done

CMD="$CMD -- neo4j"

docker run --rm \
  -v "$EXPORT_DIR:/import" \
  -v "$VOLUME_NAME:/data" \
  neo4j:5 bash -c "$CMD"
```

Expected output: `Imported: 47031 nodes, 2250197 relationships, 6799401 properties`

## 3. Start Neo4j 5 Container

```bash
docker run -d \
  --name autodistil-kg-neo4j \
  -p 7475:7474 -p 7688:7687 \
  -v autodistil-kg_neo4j_data:/data \
  -e NEO4J_AUTH=none \
  -e NEO4J_PLUGINS='["apoc"]' \
  --restart unless-stopped \
  neo4j:5
```

### Connection Details

| Setting    | Value                    |
|------------|--------------------------|
| Bolt URI   | `bolt://localhost:7688`  |
| HTTP       | `http://localhost:7475`  |
| Auth       | Disabled (`NEO4J_AUTH=none`) |
| Database   | `neo4j` (default)        |
| Neo4j Browser | http://localhost:7475/browser/ |

## 4. Verify the Import

Open Neo4j Browser at http://localhost:7475/browser/ or run:

```bash
curl -s -H "Content-Type: application/json" \
  -d '{"statements":[{"statement":"MATCH (n) RETURN labels(n)[0] AS label, count(*) AS cnt ORDER BY cnt DESC"}]}' \
  http://localhost:7475/db/neo4j/tx/commit | python3 -m json.tool
```

### Expected Node Counts

| Label              | Count  |
|--------------------|--------|
| Gene               | 20,945 |
| BiologicalProcess  | 11,381 |
| SideEffect         | 5,734  |
| MolecularFunction  | 2,884  |
| Pathway            | 1,822  |
| Compound           | 1,552  |
| CellularComponent  | 1,391  |
| Symptom            | 438    |
| Anatomy            | 402    |
| PharmacologicClass | 345    |
| Disease            | 137    |
| **Total**          | **47,031** |

### Relationship Types (24 total, 2,250,197 edges)

Key relationship types:

| Relationship          | Pattern                    | Count   |
|-----------------------|----------------------------|---------|
| PARTICIPATES_GpBP     | Gene → BiologicalProcess   | 559,504 |
| EXPRESSES_AeG         | Anatomy → Gene             | 526,407 |
| REGULATES_GrG         | Gene → Gene                | 265,672 |
| INTERACTS_GiG         | Gene ↔ Gene                | 147,164 |
| CAUSES_CcSE           | Compound → SideEffect      | 138,944 |
| ASSOCIATES_DaG        | Disease ↔ Gene             | 12,623  |
| TREATS_CtD            | Compound → Disease         | 755     |

## 5. Autodistil-KG Configuration

### API / Pipeline Config

Set these in the pipeline configuration UI or environment:

```
NEO4J_URI=bolt://localhost:7688
NEO4J_USER=neo4j
NEO4J_PASSWORD=
NEO4J_DATABASE=neo4j
```

### Recommended Graph Filters (300-500 node subgraph)

For a well-connected neuro-metabolic subgraph suitable for dataset generation:

**Node Labels:**
```
Disease, Gene, Compound, Symptom
```

**Relationship Types:**
```
ASSOCIATES_DaG, INTERACTS_GiG, TREATS_CtD, PALLIATES_CpD, PRESENTS_DpS, RESEMBLES_DrD, BINDS_CbG, UPREGULATES_DuG, DOWNREGULATES_DdG
```

**Seed Node IDs** (8 neuro-metabolic diseases):

| Disease                  | Element ID |
|--------------------------|------------|
| Schizophrenia            | `4:dbc36dbe-82fc-4fab-8193-d90d9118d382:23039` |
| Bipolar disorder         | `4:dbc36dbe-82fc-4fab-8193-d90d9118d382:22950` |
| Epilepsy syndrome        | `4:dbc36dbe-82fc-4fab-8193-d90d9118d382:22947` |
| Autistic disorder        | `4:dbc36dbe-82fc-4fab-8193-d90d9118d382:23043` |
| Obesity                  | `4:dbc36dbe-82fc-4fab-8193-d90d9118d382:22957` |
| Type 2 diabetes mellitus | `4:dbc36dbe-82fc-4fab-8193-d90d9118d382:23046` |
| Alzheimer's disease      | `4:dbc36dbe-82fc-4fab-8193-d90d9118d382:22982` |
| Parkinson's disease      | `4:dbc36dbe-82fc-4fab-8193-d90d9118d382:23006` |

Set **max_nodes** to 400-500 to stay within the target range.

> **Note:** Seed node element IDs are specific to this Neo4j instance. If you reimport the data, the element IDs will change and you'll need to look them up again by disease name.

## Backup: Original Hetionet Container

The original Hetionet container (Neo4j 3.5.12) can be run alongside as a backup:

```bash
docker run -d --name hetionet-container \
  -p 7474:7474 -p 7687:7687 \
  -v /home/bumblebee/neo4j/hetionet-data:/data \
  -v /home/bumblebee/neo4j/hetionet-logs:/logs \
  dhimmel/hetionet
```

This uses ports 7474/7687 and does not conflict with Neo4j 5 on 7475/7688.

## Re-exporting from Hetionet

If you need to regenerate the CSV export files (e.g. after losing `/tmp/hetionet-export/`):

1. Ensure the Hetionet container is running on port 7474
2. Run the export script:

```bash
python3 /tmp/hetionet-export/export_hetionet.py
```

This exports all nodes as `nodes_{Label}.csv` and all relationships as `rels_{Type}.csv` to `/tmp/hetionet-export/`.

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `Unable to retrieve routing information` | Using `neo4j://` URI scheme | Change to `bolt://localhost:7688` |
| `Database name parameter not supported in Bolt Protocol 3.0` | Connected to old Hetionet (port 7687) | Use port **7688** for Neo4j 5 |
| `Variable length relationships must not use relationship type expressions` | Neo4j 5 syntax change | Relationship types must use `\|` separator, not `:` — fixed in neo4j_provider.py |
| `Invalid credential` | Auth mismatch | Container started with `NEO4J_AUTH=none`, leave password empty |
