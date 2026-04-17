"""Neo4j graph store using the neo4j 6.x driver directly.

Replaces llama-index-graph-stores-neo4j (which pins neo4j<6 and Python<3.12)
with a thin wrapper around the neo4j driver already installed by the core
package (neo4j >=6.1.0).

The LlamaIndex TextToCypherRetriever only calls two methods on the store:
    structured_query(query, param_map)  →  list[dict]
    get_schema_str()                    →  str
plus reads the ``structured_schema`` attribute.  Neo4jDirectStore implements
exactly these — no PropertyGraphStore subclassing required.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from .config import Neo4jConfig

logger = logging.getLogger(__name__)

# Minimum Neo4j server version for vector similarity functions
NEO4J_VECTOR_MIN_VERSION = (5, 11)


def _suppress_notification_filtering() -> None:
    """Monkey-patch the neo4j driver to skip notification-filter kwargs.

    Neo4j Python driver >=5.15 passes ``notifications_*`` kwargs by default,
    which fail on Neo4j servers <5.7 (Bolt 3.0) with:
        "Notification filtering is not supported for the Bolt Protocol 3.0"
    """
    try:
        import neo4j

        _orig = neo4j.GraphDatabase.driver.__func__  # type: ignore[attr-defined]

        @classmethod  # type: ignore[misc]
        def _patched(cls, uri, **kwargs):
            kwargs.pop("notifications_min_severity", None)
            kwargs.pop("notifications_disabled_categories", None)
            kwargs.pop("notifications_disabled_classifications", None)
            return _orig(cls, uri, **kwargs)

        neo4j.GraphDatabase.driver = _patched
    except Exception:
        pass


class Neo4jDirectStore:
    """Minimal Neo4j store compatible with LlamaIndex's retriever interface.

    Uses the neo4j Python driver 6.x directly, so it works alongside
    autodistil-kg-core which already requires neo4j >=6.1.0.

    Methods expected by TextToCypherRetriever / _RetryTextToCypherRetriever:
        structured_query(query, param_map) → list[dict]
        get_schema_str(refresh)            → str
        structured_schema                  (dict attribute)
        supports_structured_queries        (bool class attribute)
    """

    supports_structured_queries: bool = True

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        database: Optional[str] = "neo4j",
    ) -> None:
        import neo4j

        _suppress_notification_filtering()
        self._driver = neo4j.GraphDatabase.driver(uri, auth=(user, password))
        self._database: Optional[str] = database
        self.structured_schema: Dict[str, Any] = {}
        self._schema_str: str = ""

        # Probe — fall back to no-database mode for Bolt 3.0 servers
        self._probe()

    def _probe(self) -> None:
        try:
            with self._driver.session(database=self._database) as s:
                s.run("RETURN 1").consume()
            logger.info("Neo4j connection OK (database=%s)", self._database)
        except Exception as exc:
            s = str(exc)
            if "Database name" in s and "Bolt Protocol" in s:
                logger.warning("Bolt 3.0 detected — disabling database selection")
                self._database = None
                with self._driver.session() as sess:
                    sess.run("RETURN 1").consume()
            else:
                raise

    def _session(self):
        if self._database is not None:
            return self._driver.session(database=self._database)
        return self._driver.session()

    def structured_query(
        self,
        query: str,
        param_map: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Execute Cypher and return results as a list of plain dicts."""
        with self._session() as session:
            result = session.run(query, param_map or {})
            return [record.data() for record in result]

    def get_schema_str(self, refresh: bool = False) -> str:
        """Return a human-readable schema string, building it on first call."""
        if not self._schema_str or refresh:
            self._refresh_schema()
        return self._schema_str

    def _refresh_schema(self) -> None:
        try:
            labels = [
                r["label"]
                for r in self.structured_query(
                    "CALL db.labels() YIELD label RETURN label"
                )
            ]
            rel_types = [
                r["relationshipType"]
                for r in self.structured_query(
                    "CALL db.relationshipTypes() YIELD relationshipType "
                    "RETURN relationshipType"
                )
            ]

            self.structured_schema = {
                "node_props": {lbl: [] for lbl in labels},
                "rel_props": {rt: [] for rt in rel_types},
                "relationships": [],
            }

            for rt in rel_types[:20]:
                try:
                    rows = self.structured_query(
                        f"MATCH (a)-[r:`{rt}`]->(b) "
                        "RETURN DISTINCT labels(a)[0] AS src, labels(b)[0] AS tgt "
                        "LIMIT 3"
                    )
                    for row in rows:
                        if row.get("src") and row.get("tgt"):
                            self.structured_schema["relationships"].append(
                                {"start": row["src"], "type": rt, "end": row["tgt"]}
                            )
                except Exception:
                    pass

            lines = [
                f"Node labels: {', '.join(labels)}",
                f"Relationship types: {', '.join(rel_types)}",
                "Relationships (source)-[type]->(target):",
                *[
                    f"  ({r['start']})-[{r['type']}]->({r['end']})"
                    for r in self.structured_schema["relationships"]
                ],
            ]
            self._schema_str = "\n".join(lines)
            logger.info(
                "Schema: %d labels, %d rel types, %d patterns",
                len(labels), len(rel_types),
                len(self.structured_schema["relationships"]),
            )
        except Exception as exc:
            logger.warning("Failed to build schema: %s", exc)
            self._schema_str = "Schema unavailable"

    def close(self) -> None:
        try:
            self._driver.close()
        except Exception:
            pass


def _detect_neo4j_version(store: Neo4jDirectStore) -> Tuple[int, ...]:
    try:
        result = store.structured_query(
            "CALL dbms.components() YIELD versions RETURN versions[0] AS ver"
        )
        if result:
            ver_str = result[0].get("ver", "0.0.0")
            return tuple(int(p) for p in ver_str.split(".")[:3])
    except Exception as exc:
        logger.debug("Could not detect Neo4j version: %s", exc)
    return (0, 0, 0)


def create_graph_store(config: Neo4jConfig) -> Tuple[Neo4jDirectStore, bool]:
    """Connect to Neo4j and return (store, supports_vector)."""
    logger.info("Connecting to Neo4j at %s (database=%s)", config.uri, config.database)

    store = Neo4jDirectStore(
        uri=config.uri,
        user=config.user,
        password=config.password,
        database=config.database,
    )

    version = _detect_neo4j_version(store)
    supports_vector = version >= NEO4J_VECTOR_MIN_VERSION

    store._refresh_schema()

    logger.info(
        "Neo4j DirectStore ready (version=%s, vector_support=%s)",
        ".".join(str(v) for v in version) if version != (0, 0, 0) else "unknown",
        supports_vector,
    )
    return store, supports_vector
