import itertools
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
    edges = set()
    for relation in relations:
        source, target, rel_type = (
            relation["source"],
            relation["target"],
            relation["relation"],
        )
        if rel_type == "<":    
            edges.add((source, target))
        elif rel_type == ">":
            edges.add((target, source))
        elif rel_type is None:
            continue
        elif rel_type == "=":
            continue
        else:
            raise ValueError(f"Unknown relation type: {rel_type}")
    
    # Drop edges that are inverse of each other
    edges = set(edge for edge in edges if (edge[1], edge[0]) not in edges)    
    
    graph = nx.DiGraph()
    graph.add_edges_from(edges)
    
    # Infer relations using POINT_TRANSITIONS
    inferred_relations = set()
    for source in graph.nodes():
        for target in nx.dfs_preorder_nodes(graph, source=source):
            if source == target:
                continue
            # check if there is a path between source and target
            if nx.has_path(graph, source, target):   
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
