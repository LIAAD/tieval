import warnings
from collections import defaultdict
from typing import List, Set, Tuple, TypedDict

import networkx as nx

from tieval.links import TLink
from tieval.temporal_relation import INVERSE_POINT_RELATION, TemporalRelation


class _DictRelation(TypedDict):
    source: str
    target: str
    relation: str


class UnionFind:
    def __init__(self):
        self.parent = {}
        self.rank = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
            return x

        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return

        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1

    def get_groups(self):
        groups = {}
        for x in self.parent:
            root = self.find(x)
            if root not in groups:
                groups[root] = []
            groups[root].append(x)
        return list(groups.values())


def _compute_point_temporal_closure(relations):
    """
    Compute the temporal closure of a set of temporal relations.

    This algorithm performs the following steps:
    1. Create a directed graph from the input relations.
    2. Infer new relations by traversing the graph and applying transition rules.
    3. Combine inferred relations with single-hop relations.

    The algorithm uses depth-first search (DFS) to explore paths between nodes
    and applies the transition rules defined in POINT_TRANSITIONS to infer
    new relations along these paths.

    Args:
        relations (List[_DictRelation]): A list of dictionaries representing
            temporal relations. Each dictionary contains 'source', 'target',
            and 'relation' keys.

    Returns:
        List[_DictRelation]: A list of dictionaries representing the temporal
        closure, including both original and inferred relations. If there is
        some inconsistency in the annotation, the function will return an empty
        list.

    Note:
        - The '>' relation is converted to '<' by swapping source and target.
        - The algorithm handles three types of relations: '<', '=', and None.
        - Inferred relations are determined using the POINT_TRANSITIONS table.
    """
    # Group equal nodes
    uf = UnionFind()
    for relation in relations:
        if relation["relation"] == "=":
            uf.union(relation["source"], relation["target"])

    grouped_equal_nodes = uf.get_groups()

    # Replace equal nodes with new nodes
    equal_node_map = {"".join(sorted(group)): group for group in grouped_equal_nodes}
    node2groupnode = {
        node: groupid for groupid, group in equal_node_map.items() for node in group
    }

    # Pre-collect all nodes for graph initialization
    nodes = set()
    edges = set()
    for relation in relations:
        source = node2groupnode.get(relation["source"], relation["source"])
        target = node2groupnode.get(relation["target"], relation["target"])
        nodes.add(source)
        nodes.add(target)

        rel_type = relation["relation"]
        if rel_type == "<":
            edges.add((source, target))
        elif rel_type == ">":
            edges.add((target, source))

    # Drop edges that are inverse of each other
    edges = {edge for edge in edges if (edge[1], edge[0]) not in edges}

    # Initialize graph with pre-known size
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)

    # Compute transitive closure
    try:
        closure = nx.transitive_closure_dag(
            graph
        )  # Use DAG-specific algorithm since temporal relations form a DAG
    except nx.exception.NetworkXUnfeasible:
        warnings.warn(
            "There is some issue in the annotation. Temporal graph is not a DAG"
        )
        return []

    inferred_relations = set()
    processed_pairs = set()
    for node1, node2 in closure.edges():
        if (node1, node2) in processed_pairs:
            continue

        processed_pairs.add((node1, node2))
        if node1 in equal_node_map and node2 in equal_node_map:
            for original_node1 in equal_node_map[node1]:
                for original_node2 in equal_node_map[node2]:
                    inferred_relations.add((original_node1, "<", original_node2))
        elif node1 in equal_node_map:
            for original_node in equal_node_map[node1]:
                inferred_relations.add((original_node, "<", node2))
        elif node2 in equal_node_map:
            for original_node in equal_node_map[node2]:
                inferred_relations.add((node1, "<", original_node))
        else:
            inferred_relations.add((node1, "<", node2))

    # Add equal relations
    for group in grouped_equal_nodes:
        for i, node1 in enumerate(group):
            for node2 in group[i + 1 :]:
                inferred_relations.add((node1, "=", node2))

    return [
        {"source": source, "target": target, "relation": relation}
        for source, relation, target in inferred_relations
    ]


def temporal_closure(tlinks: Set[TLink]) -> Set[TLink]:
    """Compute temporal closure from a set of temporal links.

    This function infers all possible TLinks form the set of tlinks
    that is fed as input.

    :param Set[TLink] tlinks:  A set of temporal links (typically from a document)
    """
    edges_triplets = tlinks_to_point_relations(tlinks)
    inferred_point_relations = _compute_point_temporal_closure(edges_triplets)
    inferred_tlinks = point_relations_to_tlinks(inferred_point_relations)
    return inferred_tlinks


def point_temporal_closure(tlinks: Set[TLink]):
    """Compute temporal closure from a set of temporal relations.

    This function infers all possible TLinks form the set of relations
    that is fed as input.
    """
    edges_triplets = tlinks_to_point_relations(tlinks)
    inferred_point_relations = _compute_point_temporal_closure(edges_triplets)
    return inferred_point_relations


def _structure_point_relation(
    source: str, target: str, point_relations: List[_DictRelation]
) -> Tuple[str, str, str, str]:
    """Map the point relations to the original structure."""
    point_relations_map = {}
    duplicate_relations = set()
    for point_relation in point_relations:
        pr_src, pr_tgt = point_relation["source"], point_relation["target"]
        relation = point_relation["relation"]
        if (pr_src, pr_tgt) not in point_relations_map:
            point_relations_map[(pr_src, pr_tgt)] = relation
        else:
            if relation != point_relations_map[(pr_src, pr_tgt)]:
                duplicate_relations.add((pr_src, pr_tgt))

        if (pr_tgt, pr_src) not in point_relations_map:
            point_relations_map[(pr_tgt, pr_src)] = INVERSE_POINT_RELATION[relation]
        else:
            if relation != point_relations_map[(pr_tgt, pr_src)]:
                duplicate_relations.add((pr_tgt, pr_src))

    # remove duplicate relations
    for relation in duplicate_relations:
        point_relations_map.pop(relation)

    # structure the point relations
    xs_ys = point_relations_map.get((f"s{source}", f"s{target}"), None)
    xs_ye = point_relations_map.get((f"s{source}", f"e{target}"), None)
    xe_ys = point_relations_map.get((f"e{source}", f"s{target}"), None)
    xe_ye = point_relations_map.get((f"e{source}", f"e{target}"), None)
    return xs_ys, xs_ye, xe_ys, xe_ye


def point_relations_to_tlinks(point_relations: List[_DictRelation]) -> Set[TLink]:
    # aggregate the point relations by entity pairs
    tlinks_point_relations = defaultdict(list)
    for point_relation in point_relations:
        key = tuple(
            sorted((point_relation["source"][1:], point_relation["target"][1:]))
        )
        tlinks_point_relations[key].append(point_relation)

    # assert if the point relations found form a valid interval relation
    inferred_tlinks = []
    for (source, target), point_relation in tlinks_point_relations.items():
        xs_ys, xs_ye, xe_ys, xe_ye = _structure_point_relation(
            source, target, point_relation
        )

        relation = TemporalRelation([xs_ys, xs_ye, xe_ys, xe_ye])
        if relation.is_complete():
            tlink = TLink(source, target, relation)
            inferred_tlinks.append(tlink)
    return set(inferred_tlinks)


def tlinks_to_point_relations(tlinks: Set[TLink]) -> List[_DictRelation]:
    point_relations = []
    for tlink in tlinks:
        sx = f"s{tlink.source_id}"
        ex = f"e{tlink.source_id}"
        sy = f"s{tlink.target_id}"
        ey = f"e{tlink.target_id}"

        p_relations = tlink.relation.point.relation

        point_relations += [
            {"source": sx, "target": sy, "relation": p_relations[0]},
            {"source": sx, "target": ey, "relation": p_relations[1]},
            {"source": ex, "target": sy, "relation": p_relations[2]},
            {"source": ex, "target": ey, "relation": p_relations[3]},
            {"source": sx, "target": ex, "relation": "<"},
            {"source": sy, "target": ey, "relation": "<"},
        ]

    return point_relations
