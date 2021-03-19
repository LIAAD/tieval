import collections

from typing import Tuple
from typing import List
from typing import Set

import pandas as pd

from tqdm import tqdm

# constants representing the start point and end point of an interval
_start = 0
_end = 1
_src = 0  # source
_tgt = 1  # target

# mapping from interval relation names to point relations
# for example, BEFORE means that the first interval's end is before the second interval's start
_interval_to_point = {
    "BEFORE": [(_src, _end, "<", _tgt, _start)],
    "AFTER": [(_src, _start, '>', _tgt, _end)],
    "IBEFORE": [(_src, _end, "=", _tgt, _start)],
    "IAFTER": [(_src, _start, "=", _tgt, _end)],
    # "CONTAINS": [(_src, _start, "<", _tgt, _start), (_tgt, _end, "<", _src, _end)],
    "INCLUDES": [(_src, _start, "<", _tgt, _start), (_src, _end, '>', _tgt, _end)],
    "IS_INCLUDED": [(_src, _start, '>', _tgt, _start), (_src, _end, "<", _tgt, _end)],
    "BEGINS-ON": [(_src, _start, "=", _tgt, _start)],
    "ENDS-ON": [(_src, _end, "=", _tgt, _end)],
    "BEGINS": [(_src, _start, "=", _tgt, _start), (_src, _end, "<", _tgt, _end)],
    "BEGUN_BY": [(_src, _start, "=", _tgt, _start), (_src, _end, '>', _tgt, _end)],
    "ENDS": [(_src, _start, '>', _tgt, _start), (_src, _end, "=", _tgt, _end)],
    "ENDED_BY": [(_src, _start, "<", _tgt, _start), (_src, _end, "=", _tgt, _end)],
    "SIMULTANEOUS": [(_src, _start, "=", _tgt, _start), (_src, _end, "=", _tgt, _end)],
    # "IDENTITY": [(_src, _start, "=", _tgt, _start), (_src, _end, "=", _tgt, _end)],
    # "DURING": [(_src, _start, "=", _tgt, _start), (_src, _end, "=", _tgt, _end)],
    # "DURING_INV": [(_src, _start, "=", _tgt, _start), (_src, _end, "=", _tgt, _end)],
    "OVERLAP": [(_src, _start, "<", _tgt, _end), (_src, _end, '>', _tgt, _start)],
}
# transitivity table for point relations
_point_transitions = {
    "<": {"<": "<", "=": "<"},
    "=": {"<": "<", "=": "="},
}

relevant_relations = {
    'OVERLAP': 'OVERLAP',
    'BEGINS': 'BEGINS',
    'BEFORE': 'BEFORE',
    'CONTAINS': 'INCLUDES',
    'IDENTITY': 'SIMULTANEOUS',
    'AFTER': 'AFTER',
    'BEGINS-ON': 'BEGINS-ON',
    'SIMULTANEOUS': 'SIMULTANEOUS',
    'INCLUDES': 'INCLUDES',
    'DURING': 'SIMULTANEOUS',
    'ENDS-ON': 'ENDS-ON',
    'BEGUN_BY': 'BEGUN_BY',
    'ENDED_BY': 'ENDED_BY',
    'DURING_INV': 'SIMULTANEOUS',
    'ENDS': 'ENDS',
    'IS_INCLUDED': 'IS_INCLUDED',
    'IBEFORE': 'IBEFORE',
    'IAFTER': 'IAFTER'
}


def remove_duplicate_relations(annotations):
    seen_point_relations = set()
    result_annotations = set()
    for annotation in annotations:

        # only include this annotation if no previous annotation expanded to the same point relations
        point_relations = frozenset(to_point_relations(annotation))
        if point_relations not in seen_point_relations:
            seen_point_relations.add(point_relations)
            result_annotations.add(annotation)

    # return the filtered annotations
    return result_annotations


def to_point_relations(annotation):
    start = _start
    end = _end

    # converts an interval relation to point relations
    point_relations = set()
    interval1, interval2, value = annotation
    intervals = (interval1, interval2)

    # the start of an interval is always before its end
    point_relations.add(((interval1, start), "<", (interval1, end)))
    point_relations.add(((interval2, start), "<", (interval2, end)))

    # use the interval-to-point lookup table to add the necessary point relations
    for index1, side1, relation, index2, side2 in _interval_to_point[value]:
        point1 = (intervals[index1], side1)
        point2 = (intervals[index2], side2)
        point_relations.add((point1, relation, point2))

        # for reflexive point relations, add them in the other direction too
        if relation == "=":
            point_relations.add((point2, relation, point1))

    # return the collected relations
    return point_relations


def to_interval_relations(point_relations):
    # find all pairs of intervals.
    pair_names = {}
    for ((interval1, _), _, (interval2, _)) in point_relations:
        pair_names[(interval1, interval2)] = None
        pair_names[(interval2, interval1)] = None

    # for each interval pair, see if it satisfies the point-wise requirements for any interval relations
    interval_relations = set()
    for pair in pair_names:
        for relation, requirements in _interval_to_point.items():
            if all(((pair[i1], s1), r, (pair[i2], s2)) in point_relations for i1, s1, r, i2, s2 in requirements):
                interval_relations.add((*pair, relation))

    # return the collected relations
    return interval_relations


def temporal_closure(annotations: Set[Tuple]):
    annotations = remove_duplicate_relations(annotations)

    # convert interval relations to point relations
    new_relations = {r for a in annotations for r in to_point_relations(a)}

    # repeatedly apply point transitivity rules until no new relations can be inferred
    point_relations = set()
    point_relations_index = collections.defaultdict(set)
    while new_relations:

        # update the result and the index with any new relations found on the last iteration
        point_relations.update(new_relations)
        for point_relation in new_relations:
            point_relations_index[point_relation[0]].add(point_relation)

        # infer any new transitive relations, e.g., if A < B and B < C then A < C
        new_relations = set()
        for point1, relation12, point2 in point_relations:
            for _, relation23, point3 in point_relations_index[point2]:
                relation13 = _point_transitions[relation12][relation23]
                new_relation = (point1, relation13, point3)
                if new_relation not in point_relations:
                    new_relations.add(new_relation)

    # convert the point relations back to interval relations
    closure_relations = to_interval_relations(point_relations)

    # remove redundant relations.
    return {(s, t, v) for s, t, v in closure_relations if s != t}


def temporal_awareness(annotation: Set[Tuple], system: Set[Tuple]) -> float:
    annotation = remove_duplicate_relations(annotation)
    system = remove_duplicate_relations(system)

    if len(system) == 0:
        raise RuntimeError("The system annotation in empty.")

    if len(annotation) == 0:
        raise RuntimeError("The reference annotation in empty.")

    precision = len(temporal_closure(annotation) & system) / len(system)
    recall = len(temporal_closure(system) & annotation) / len(annotation)

    return 2 * precision * recall / (precision + recall)


def multifile_temporal_awareness(files_annotation: List[Set[Tuple]], files_system: List[Set[Tuple]]):
    reference = 0
    predicted = 0
    precision_correct = 0
    recall_correct = 0
    for annotation, system in zip(files_annotation, files_system):
        annotation = remove_duplicate_relations(annotation)
        system = remove_duplicate_relations(system)

        reference += len(annotation)
        predicted += len(system)

        precision_correct += len(temporal_closure(annotation) & system)
        recall_correct += len(temporal_closure(system) & annotation)
    precision = precision_correct / reference
    recall = recall_correct / predicted
    return 2 * recall * precision / (precision + recall)


def get_annotations(tlinks: pd.DataFrame, features: List=['source', 'relatedTo', 'relType']) -> List[Set[Tuple]]:
    """
    Returns the annotations in the shape [{(A, B, rel1), (B, C, rel2), ...}, {(A, B, rel1), (B, C, rel2), ...}]
    given a tlinks dataframe. Each set in the list output is the annotation for a document.
    :param tlinks:
    :return:
    """
    annotations = []
    for file in tqdm(tlinks.file.unique()):
        df = tlinks.loc[tlinks.file == file, features]
        file_annotations = {tuple(row.tolist()) for _, row in df.iterrows()}
        annotations.append(file_annotations)
    return annotations


def get_temporal_closure(annotations: List[Set[Tuple]]) -> List[Set[Tuple]]:
    """
    Computes the temporal closure of the annotations in the format given by the function 'get_annotations'.
    :param annotations:
    :return:
    """
    full_closure = [temporal_closure(file_annotation) for file_annotation in tqdm(annotations)]

    # The closure can infer tlinks for pairs of events/timex that are on the gold annotations. The next block removes
    # this ambiguity.
    clean_closure = []
    for file, file_tc in zip(annotations, full_closure):
        annot_base = set((e1, e2) for e1, e2, _ in file)
        file.update({(e1, e2, rel) for e1, e2, rel in file_tc if (e1, e2) not in annot_base})
        clean_closure.append(file)
    return clean_closure
