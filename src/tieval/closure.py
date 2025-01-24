import itertools
import collections
from collections import defaultdict
from typing import List, Set, Tuple, TypedDict

import networkx as nx

from tieval.links import TLink
from tieval.temporal_relation import INVERSE_POINT_RELATION, TemporalRelation


class _DictRelation(TypedDict):
    source: str
    target: str
    relation: str


def _compute_point_temporal_closure(relations: List[_DictRelation]):
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
        closure, including both original and inferred relations.

    Note:
        - The '>' relation is converted to '<' by swapping source and target.
        - The algorithm handles three types of relations: '<', '=', and None.
        - Inferred relations are determined using the POINT_TRANSITIONS table.
    """

    # Group equal nodes
    grouped_equal_nodes = []
    for relation in relations:
        if relation["relation"] == "=":
            already_equal = False
            for node in grouped_equal_nodes:
                if relation["source"] in node:
                    node.append(relation["target"])
                    already_equal = True
                elif relation["target"] in node:
                    node.append(relation["source"])
                    already_equal = True
            if not already_equal:
                grouped_equal_nodes.append([relation["source"], relation["target"]])

    # Replace equal nodes with new nodes
    equal_node_map = {"".join(sorted(group)): group for group in grouped_equal_nodes}
    node2groupnode = {
        node: groupid for groupid, group in equal_node_map.items() for node in group
    }

    for relation in relations:
        if relation["source"] in node2groupnode:
            relation["source"] = node2groupnode[relation["source"]]
        if relation["target"] in node2groupnode:
            relation["target"] = node2groupnode[relation["target"]]

    # make all relations "<"
    before_relations = []
    graph = nx.DiGraph()
    for relation in relations:
        source, target, rel_type = (
            relation["source"],
            relation["target"],
            relation["relation"],
        )
        if rel_type == "<":
            graph.add_edge(source, target, relation=rel_type)
            before_relations.append((source, target, rel_type))
        elif rel_type == ">":
            graph.add_edge(target, source, relation="<")
            before_relations.append((target, source, "<"))
        elif rel_type is None:
            continue
        elif rel_type == "=":
            continue
        else:
            raise ValueError(f"Unknown relation type: {rel_type}")

    # Infer relations using POINT_TRANSITIONS
    inferred_relations = set()
    for source in graph.nodes():
        for target in nx.dfs_preorder_nodes(graph, source=source):
            if source == target:
                continue

            path = nx.shortest_path(graph, source, target)
            combinations = itertools.combinations(path, 2)
            for node1, node2 in combinations:
                if node1 in equal_node_map and node2 in equal_node_map:
                    for original_node1 in equal_node_map[node1]:
                        for original_node2 in equal_node_map[node2]:
                            inferred_relations.add(
                                (original_node1, "<", original_node2)
                            )
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
        combinations = itertools.combinations(group, 2)
        for node1, node2 in combinations:
            inferred_relations.add((node1, "=", node2))

    return [
        {"source": source, "target": target, "relation": relation}
        for source, relation, target in inferred_relations
    ]


def _remove_duplicate_tlinks(tlinks: Set[TLink]) -> Set[TLink]:
    """Remove duplicate tlinks from a set of tlinks.

    This function removes duplicate tlinks from a set of tlinks by
    comparing the source, target, and relation of the tlinks.
    """

    def tlink_key(tlink: TLink) -> str:
        return "".join(
            sorted(list(tlink.source.id + tlink.target.id + tlink.relation.interval))
        )

    tlink_keys = [tlink_key(tlink) for tlink in tlinks]
    key_count = collections.Counter(tlink_keys)
    tlink_keys_to_remove = [key for key, count in key_count.items() if count > 1]
    unique_tlinks = [tlink for tlink in tlinks if tlink_key(tlink) not in tlink_keys_to_remove]
    return unique_tlinks


def temporal_closure(tlinks: Set[TLink]) -> Set[TLink]:
    """Compute temporal closure from a set of temporal links.

    This function infers all possible TLinks form the set of tlinks
    that is fed as input.

    :param Set[TLink] tlinks:  A set of temporal links (typically from a document)
    """
    tlinks = _remove_duplicate_tlinks(tlinks)

    edges_triplets = tlinks_to_point_relations(tlinks)
    inferred_point_relations = _compute_point_temporal_closure(edges_triplets)
    inferred_tlinks = point_relations_to_tlinks(inferred_point_relations)
    return inferred_tlinks


def point_temporal_closure(tlinks: Set[TLink]):
    """Compute temporal closure from a set of temporal relations.

    This function infers all possible TLinks form the set of relations
    that is fed as input.
    """
    tlinks = _remove_duplicate_tlinks(tlinks)
    edges_triplets = tlinks_to_point_relations(tlinks)
    inferred_point_relations = _compute_point_temporal_closure(edges_triplets)
    inferred_tlinks = point_relations_to_tlinks(inferred_point_relations)
    return inferred_tlinks


def _structure_point_relation(
    source: str, target: str, point_relation: Set[str]
) -> Tuple[str, str, str, str]:
    """Map the point relations to the original structure."""

    xs_ys, xs_ye, xe_ys, xe_ye = None, None, None, None
    for node1, relation, node2 in point_relation:
        if node1 == f"s{source}" and node2 == f"s{target}":
            xs_ys = relation
        elif node2 == f"s{source}" and node1 == f"s{target}":
            xs_ys = INVERSE_POINT_RELATION[relation]

        elif node1 == f"s{source}" and node2 == f"e{target}":
            xs_ye = relation
        elif node2 == f"s{source}" and node1 == f"e{target}":
            xs_ye = INVERSE_POINT_RELATION[relation]

        elif node1 == f"e{source}" and node2 == f"s{target}":
            xe_ys = relation
        elif node2 == f"e{source}" and node1 == f"s{target}":
            xe_ys = INVERSE_POINT_RELATION[relation]

        elif node1 == f"e{source}" and node2 == f"e{target}":
            xe_ye = relation
        elif node2 == f"e{source}" and node1 == f"e{target}":
            xe_ye = INVERSE_POINT_RELATION[relation]

    return xs_ys, xs_ye, xe_ys, xe_ye


def point_relations_to_tlinks(point_relations: List[_DictRelation]) -> Set[TLink]:
    # aggregate the point relations by entity pairs
    tlinks_point_relations = defaultdict(set)
    for point_relation in point_relations:
        key = tuple(
            sorted((point_relation["source"][1:], point_relation["target"][1:]))
        )
        tlinks_point_relations[key].add(
            (
                point_relation["source"],
                point_relation["relation"],
                point_relation["target"],
            )
        )

    # assert if the point relations found form a valid interval relation
    inferred_tlinks = set()
    for entities, point_relation in tlinks_point_relations.items():
        source, target = sorted(entities)
        xs_ys, xs_ye, xe_ys, xe_ye = _structure_point_relation(
            source, target, point_relation
        )

        relation = TemporalRelation([xs_ys, xs_ye, xe_ys, xe_ye])
        if relation.is_complete():
            tlink = TLink(source, target, relation)
            inferred_tlinks.add(tlink)
    return inferred_tlinks


def tlinks_to_point_relations(tlinks: Set[TLink]) -> List[_DictRelation]:
    point_relations = []
    for tlink in tlinks:
        if isinstance(tlink.source, str):
            sx = f"s{tlink.source}"
            ex = f"e{tlink.source}"
            sy = f"s{tlink.target}"
            ey = f"e{tlink.target}"

        else:
            sx = f"s{tlink.source.id}"
            ex = f"e{tlink.source.id}"
            sy = f"s{tlink.target.id}"
            ey = f"e{tlink.target.id}"

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
