from collections import defaultdict
import copy
from typing import Set

from tieval.links import TLink


def temporal_closure(
        tlinks: Set[TLink],
        allow_incomplete: bool = False
) -> Set[TLink]:
    """Compute temporal closure from a set of temporal links.

    This function infers all possible TLinks form the set of tlinks
    that is fed as input.

    :param Set[TLink] tlinks:  A set of temporal links (typically from a document)
    :param bool allow_incomplete: An incomplete TLink is one in which the temporal relation between the
        endpoints of source and target is not totally defined.
        That is, one of the point relations is None. By default, this parameter
        is set to False.

    :returns: The union of the tlinks given as input and the tlinks inferred.
    :rtype: Set[TLink]

    .. seealso::
        Check this `paper`_ to learn more about temporal closure.

    .. _paper: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.468.2433&rep=rep1&type=pdf
    """

    result = copy.deepcopy(tlinks)

    # build a dictionary of entities and their tlinks
    entities_tlinks = defaultdict(set)
    for tlink in tlinks:
        entities_tlinks[tlink.source].add(tlink)
        entities_tlinks[tlink.target].add(tlink)

    old_tlinks = tlinks
    while True:

        inferred_tlinks = set()
        for tlink1 in old_tlinks:

            connected_tlinks = set.union(
                entities_tlinks[tlink1.source],
                entities_tlinks[tlink1.target]
            )
            connected_tlinks.discard(tlink1)

            for tlink2 in connected_tlinks:

                inferred_tlink = tlink1 & tlink2

                if inferred_tlink is not None:
                    continue

                # check if the inferred tlink is novel
                cond = (inferred_tlink in result)
                if not allow_incomplete:
                    cond = cond and inferred_tlink.relation.is_complete()

                if cond:

                    inferred_tlinks.add(inferred_tlink)

                    # update entities_tlinks dict
                    entities_tlinks[inferred_tlink.source].add(inferred_tlink)
                    entities_tlinks[inferred_tlink.target].add(inferred_tlink)

        if inferred_tlinks:
            result.update(inferred_tlinks)
            old_tlinks = inferred_tlinks

        else:
            break

    # remove inferred tlinks that are an incomplete inference of existing ones.
    # example: A --BEFORE--> B and A --BEFORE-OR-OVERLAP--> B
    to_remove = []
    entity_pairs = {}
    for tlink in result:
        key = tuple(sorted([tlink.source.id, tlink.target.id]))

        if key not in entity_pairs:
            entity_pairs[key] = tlink

        else:
            prev_tlink = entity_pairs[key]

            if tlink.source != prev_tlink.source:
                prev_tlink = ~prev_tlink

            print(f"{tlink}\n{prev_tlink}")

            # check if one point relation is contained in the other
            # example: point relation [<, <, None, <] is contained in [<, <, <, <]
            # an example that would not be removed: ['<', '<', '>', '>'] and ['>', '>', '>', '>']
            # we don't what this links to be removed as they expose errors in the annotation
            is_contained = True
            for p1, p2 in zip(prev_tlink.relation.point, tlink.relation.point):

                if p1 is None or p2 is None:
                    continue

                elif p1 != p2:
                    is_contained = False
                    break

            if is_contained:
                to_remove += [tlink]

    return result
