from collections import defaultdict
from typing import Set, Tuple

import networkx as nx

from tieval.links import TLink
from tieval.temporal_relation import _INVERSE_POINT_RELATION, TemporalRelation


def temporal_closure(tlinks: Set[TLink]) -> Set[TLink]:
    """Compute temporal closure from a set of temporal links.

    This function infers all possible TLinks form the set of tlinks
    that is fed as input.

    :param Set[TLink] tlinks:  A set of temporal links (typically from a document)
ep1&type=pdf
    """

    edges_triplets = set()
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

        edges_triplets.update([
            (sx, p_relations[0], sy),
            (sx, p_relations[1], ey),
            (ex, p_relations[2], sy),
            (ex, p_relations[3], ey)
        ])

    # make all relations "<" or "="
    edges = set()
    equal_nodes = set()
    for node1, relation, node2 in edges_triplets:

        if relation == "<":
            edges.add((node1, node2))

        if relation == ">":
            edges.add((node2, node1))

        elif relation == "=":
            sorted_nodes = tuple(sorted((node1, node2)))
            equal_nodes.add(sorted_nodes)

    # build to equal graph
    equal_graph = nx.Graph()
    equal_graph.add_edges_from(equal_nodes)

    equal_point_relations = set()
    for connected_nodes in nx.connected_components(equal_graph):
        while len(connected_nodes) > 1:
            n1 = connected_nodes.pop()
            for n2 in connected_nodes:
                equal_point_relations.add((n1, "=", n2))

    # build temporal graph
    tempgraph = nx.DiGraph()
    tempgraph.add_edges_from(edges)
    inferred_point_relations = _get_connected_nodes(tempgraph)
    inferred_point_relations = set((n1, "<", n2) for n1, n2 in inferred_point_relations)

    # add relations to points that are equivalent
    new_point_relations = set()
    for relation in inferred_point_relations:
        for node1_eq, _, node2_eq in equal_point_relations:

            if node2_eq in relation:
                new_point_relations.update([tuple(map(lambda x: x.replace(node2_eq, node1_eq), relation))])

            if node1_eq in relation:
                new_point_relations.update([tuple(map(lambda x: x.replace(node1_eq, node2_eq), relation))])

    inferred_point_relations.update(new_point_relations)
    inferred_point_relations.update(equal_point_relations)

    # aggregate the point relations by entity pairs
    tlinks_point_relations = defaultdict(set)
    for source, relation, target in inferred_point_relations:
        key = tuple(sorted((source[1:], target[1:])))
        tlinks_point_relations[key].add((source, relation, target))

    # assert if the point relations found form a valid interval relation
    inferred_tlinks = set()
    for entities, point_relation in tlinks_point_relations.items():

        source, target = sorted(entities)
        xs_ys, xs_ye, xe_ys, xe_ye = _structure_point_relation(source, target, point_relation)

        relation = TemporalRelation([xs_ys, xs_ye, xe_ys, xe_ye])
        if relation.is_complete():
            tlink = TLink(source, target, relation)
            inferred_tlinks.add(tlink)

    return inferred_tlinks


def _get_connected_nodes(graph: nx.Graph) -> Set[Tuple[str, str]]:
    """Retrieve the pairs of nodes that are connected by a path

    :param graph: A directed graph.
    :type: nx.Graph

    :return: Set[Tuple[str, str]]
    """

    # segment the temporal graph in disconnected graphs
    undirected = graph.to_undirected()
    sub_graphs_nodes = nx.connected_components(undirected)
    sub_graphs = [graph.subgraph(nodes) for nodes in sub_graphs_nodes]

    # retrieve all the possible paths between root and leaf nodes
    node_pairs = set()
    for sub_graph in sub_graphs:

        for node in sub_graph.nodes:
            descendants = nx.algorithms.descendants(sub_graph, node)
            node_pairs.update(list(zip([node] * len(descendants), descendants)))

    return node_pairs


def _structure_point_relation(
        source: str,
        target: str,
        point_relation: Set[str]
) -> Tuple[str, str, str, str]:
    """Map the point relations to the original structure.
    """

    xs_ys, xs_ye, xe_ys, xe_ye = None, None, None, None
    for node1, relation, node2 in point_relation:

        if node1 == f"s{source}" and node2 == f"s{target}":
            xs_ys = relation
        elif node2 == f"s{source}" and node1 == f"s{target}":
            xs_ys = _INVERSE_POINT_RELATION[relation]

        elif node1 == f"s{source}" and node2 == f"e{target}":
            xs_ye = relation
        elif node2 == f"s{source}" and node1 == f"e{target}":
            xs_ye = _INVERSE_POINT_RELATION[relation]

        elif node1 == f"e{source}" and node2 == f"s{target}":
            xe_ys = relation
        elif node2 == f"e{source}" and node1 == f"s{target}":
            xe_ys = _INVERSE_POINT_RELATION[relation]

        elif node1 == f"e{source}" and node2 == f"e{target}":
            xe_ye = relation
        elif node2 == f"e{source}" and node1 == f"e{target}":
            xe_ye = _INVERSE_POINT_RELATION[relation]

    return xs_ys, xs_ye, xe_ys, xe_ye
