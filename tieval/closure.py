import copy
from typing import Set
from collections import defaultdict

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
    for tl in tlinks:
        entities_tlinks[tl.source].add(tl)
        entities_tlinks[tl.target].add(tl)

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

                # check if the inferred tlink is novel
                already_inferred = (inferred_tlink in result)
                cond = inferred_tlink and not already_inferred
                if not allow_incomplete:
                    cond = cond and inferred_tlink.relation.is_complete()

                if cond:

                    if inferred_tlink.source == "L" and inferred_tlink.target == "K":
                        print()

                    inferred_tlinks.add(inferred_tlink)

                    # update entities_tlinks dict
                    entities_tlinks[inferred_tlink.source].add(inferred_tlink)
                    entities_tlinks[inferred_tlink.source].add(inferred_tlink)

        if inferred_tlinks:
            result.update(inferred_tlinks)
            old_tlinks = inferred_tlinks

        else:
            break

    return result
