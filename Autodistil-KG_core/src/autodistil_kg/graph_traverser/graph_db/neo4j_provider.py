"""Neo4j graph database implementation."""
from typing import Dict, List, Optional, Any
from neo4j import GraphDatabase as _neo4j_driver, Driver
import logging

from .interface import GraphDatabase

logger = logging.getLogger(__name__)

# Reduce Neo4j deprecation/notification log spam (id→elementId, property warnings)
logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)


class Neo4jGraphDatabase(GraphDatabase):
    """Neo4j implementation of the GraphDatabase interface."""

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        database: Optional[str] = None
    ):
        """
        Initialize Neo4j connection.

        Args:
            uri: Neo4j connection URI (e.g., "bolt://localhost:7687")
            user: Neo4j username
            password: Neo4j password
            database: Optional database name (defaults to default database)
        """
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.driver: Optional[Driver] = None
        self._has_element_id = False  # True when Neo4j >= 4.4

    def connect(self) -> None:
        """Establish connection to Neo4j."""
        try:
            self.driver = _neo4j_driver.driver(
                self.uri,
                auth=(self.user, self.password)
            )
            # Verify connectivity
            self.driver.verify_connectivity()
            logger.info(f"Connected to Neo4j at {self.uri}")
            self._detect_element_id_support()
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise

    def _detect_element_id_support(self) -> None:
        """Detect whether the server supports elementId() (Neo4j >= 4.4)."""
        try:
            with self._get_session() as session:
                session.run("MATCH (n) RETURN elementId(n) LIMIT 0").consume()
            self._has_element_id = True
            logger.debug("Neo4j server supports elementId()")
        except Exception:
            self._has_element_id = False
            logger.debug("Neo4j server does not support elementId(), using id()")

    def _node_id_expr(self, var: str = "n") -> str:
        """Return the Cypher expression for getting a node's stable ID."""
        if self._has_element_id:
            return f"elementId({var})"
        return f"toString(id({var}))"

    def _node_id_match(self, var: str, param: str, node_id: str) -> str:
        """Return a WHERE clause fragment to match a node by ID."""
        if self._has_element_id and ":" in (node_id or ""):
            return f"elementId({var}) = ${param}"
        return f"id({var}) = toInteger(${param})"
    
    def disconnect(self) -> None:
        """Close Neo4j connection."""
        if self.driver:
            self.driver.close()
            self.driver = None
            logger.info("Disconnected from Neo4j")
    
    def _get_session(self):
        """Get a Neo4j session, omitting database param for older servers."""
        if not self.driver:
            raise RuntimeError("Not connected to Neo4j. Call connect() first.")
        if self.database:
            try:
                return self.driver.session(database=self.database)
            except TypeError:
                # Very old driver versions may not accept database kwarg
                return self.driver.session()
        return self.driver.session()
    
    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a node by its ID (elementId or legacy integer id)."""
        id_expr = self._node_id_expr("n")
        where = self._node_id_match("n", "node_id", node_id)
        query = f"MATCH (n) WHERE {where} RETURN n, labels(n) as labels, {id_expr} as eid LIMIT 1"

        with self._get_session() as session:
            result = session.run(query, node_id=node_id)
            record = result.single()

            if record:
                node = dict(record["n"])
                eid = record.get("eid")
                return {
                    "id": str(eid) if eid is not None else node_id,
                    "labels": record["labels"],
                    "properties": node
                }
            return None
    
    def get_neighbors(
        self,
        node_id: str,
        relationship_types: Optional[List[str]] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get neighboring nodes of a given node."""
        if relationship_types:
            rel_filter = f":{'|'.join(relationship_types)}"
        else:
            rel_filter = ""

        limit_clause = f"LIMIT {limit}" if limit else ""
        where_clause = self._node_id_match("n", "node_id", node_id)
        id_expr = self._node_id_expr("neighbor")

        query = f"""
        MATCH (n)-[r{rel_filter}]-(neighbor)
        WHERE {where_clause}
        RETURN neighbor, labels(neighbor) as labels, {id_expr} as neighbor_eid,
               type(r) as relationship_type, properties(r) as rel_properties
        {limit_clause}
        """

        with self._get_session() as session:
            result = session.run(query, node_id=node_id)
            neighbors = []

            for record in result:
                neighbor = dict(record["neighbor"])
                eid = record.get("neighbor_eid")
                neighbors.append({
                    "id": str(eid) if eid is not None else node_id,
                    "labels": record["labels"],
                    "properties": neighbor,
                    "relationship_type": record["relationship_type"],
                    "relationship_properties": record["rel_properties"]
                })

            return neighbors
    
    def get_node_properties(self, node_id: str) -> Dict[str, Any]:
        """Get all properties of a node."""
        where = self._node_id_match("n", "node_id", node_id)
        query = f"MATCH (n) WHERE {where} RETURN properties(n) as props LIMIT 1"

        with self._get_session() as session:
            result = session.run(query, node_id=node_id)
            record = result.single()

            if record:
                return dict(record["props"])
            return {}
    
    def get_relationships(
        self,
        node_id: str,
        relationship_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Get all relationships of a node."""
        if relationship_types:
            rel_filter = f":{'|'.join(relationship_types)}"
        else:
            rel_filter = ""

        where_clause = self._node_id_match("n", "node_id", node_id)
        start_id_expr = self._node_id_expr("startNode(r)")
        end_id_expr = self._node_id_expr("endNode(r)")

        query = f"""
        MATCH (n)-[r{rel_filter}]-(other)
        WHERE {where_clause}
        RETURN type(r) as type, properties(r) as properties,
               {start_id_expr} as start_id, {end_id_expr} as end_id,
               labels(startNode(r)) as start_labels,
               labels(endNode(r)) as end_labels
        """

        with self._get_session() as session:
            result = session.run(query, node_id=node_id)
            relationships = []

            for record in result:
                relationships.append({
                    "type": record["type"],
                    "properties": dict(record["properties"]),
                    "start_id": str(record["start_id"]),
                    "end_id": str(record["end_id"]),
                    "start_labels": record["start_labels"],
                    "end_labels": record["end_labels"]
                })

            return relationships
    
    def query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute a custom query on Neo4j."""
        params = parameters or {}
        
        with self._get_session() as session:
            result = session.run(query, **params)
            return [dict(record) for record in result]
    
    def get_subgraph(
        self,
        node_id: str,
        depth: int = 2,
        relationship_types: Optional[List[str]] = None,
        limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get the subgraph around a node up to a given depth.
        Returns center node, all nodes/edges within depth, and all paths.
        """
        if relationship_types:
            rel_filter = f":{'|'.join(relationship_types)}"
        else:
            rel_filter = ""

        where_clause = self._node_id_match("center", "node_id", node_id)
        id_expr_n = self._node_id_expr("n")
        id_expr_center = self._node_id_expr("center")
        id_expr_start = self._node_id_expr("startNode(rel)")
        id_expr_end = self._node_id_expr("endNode(rel)")

        limit_clause = f"LIMIT {limit}" if limit else "LIMIT 500"

        # Query all paths from the center node up to the given depth
        query = f"""
        MATCH path = (center)-[r{rel_filter}*1..{depth}]-(connected)
        WHERE {where_clause}
        WITH path, center, connected,
             [n IN nodes(path) | {{
                 id: {id_expr_n},
                 labels: labels(n),
                 properties: properties(n)
             }}] AS path_nodes,
             [rel IN relationships(path) | {{
                 source_id: {id_expr_start},
                 target_id: {id_expr_end},
                 type: type(rel),
                 properties: properties(rel)
             }}] AS path_rels
        RETURN path_nodes, path_rels,
               {id_expr_center} AS center_id,
               labels(center) AS center_labels,
               properties(center) AS center_props
        {limit_clause}
        """

        with self._get_session() as session:
            result = session.run(query, node_id=node_id)

            nodes_map: Dict[str, Dict[str, Any]] = {}
            edges_list: List[Dict[str, Any]] = []
            paths: List[List[Dict[str, Any]]] = []
            center_node = None
            seen_edges = set()

            for record in result:
                # Build center node on first record
                if center_node is None:
                    center_id = str(record["center_id"])
                    center_node = {
                        "id": center_id,
                        "labels": record["center_labels"],
                        "properties": dict(record["center_props"]),
                    }
                    nodes_map[center_id] = center_node

                # Collect nodes
                path_nodes = record["path_nodes"]
                for pn in path_nodes:
                    nid = str(pn["id"])
                    if nid not in nodes_map:
                        nodes_map[nid] = {
                            "id": nid,
                            "labels": list(pn["labels"]),
                            "properties": dict(pn["properties"]),
                        }

                # Collect edges (deduplicate)
                path_rels = record["path_rels"]
                for pr in path_rels:
                    edge_key = (str(pr["source_id"]), str(pr["target_id"]), pr["type"])
                    if edge_key not in seen_edges:
                        seen_edges.add(edge_key)
                        edge = {
                            "source_id": str(pr["source_id"]),
                            "target_id": str(pr["target_id"]),
                            "type": pr["type"],
                            "properties": dict(pr["properties"]) if pr["properties"] else {},
                        }
                        edges_list.append(edge)

                # Build the path as alternating nodes and edges
                path = []
                for i, pn in enumerate(path_nodes):
                    nid = str(pn["id"])
                    path.append(nodes_map[nid])
                    if i < len(path_rels):
                        pr = path_rels[i]
                        path.append({
                            "source_id": str(pr["source_id"]),
                            "target_id": str(pr["target_id"]),
                            "type": pr["type"],
                            "properties": dict(pr["properties"]) if pr["properties"] else {},
                        })
                paths.append(path)

            if center_node is None:
                # No paths found; fetch the center node directly
                center_node_data = self.get_node(node_id)
                if center_node_data:
                    center_node = center_node_data
                    nodes_map[center_node["id"]] = center_node
                else:
                    center_node = {"id": node_id, "labels": [], "properties": {}}

            return {
                "center": center_node,
                "nodes": nodes_map,
                "edges": edges_list,
                "paths": paths,
            }

    def get_node_count(self) -> int:
        """Get total number of nodes in the graph."""
        query = "MATCH (n) RETURN count(n) as count"
        
        with self._get_session() as session:
            result = session.run(query)
            record = result.single()
            return record["count"] if record else 0
