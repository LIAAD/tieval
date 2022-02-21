from typing import Set

from text2timeline.links import TLink
import copy


def temporal_closure(tlinks: Set[TLink]) -> Set[TLink]:
    """Compute temporal closure from a set of temporal links.

    This function infers all possible TLinks form the set of tlinks
    that is fed as input.

    Parameters
    ----------
    tlinks: Set[TLink]
        A set of temporal links (typically from a document)
    allow_incomplete: bool
        An incomplete TLink is one in which the temporal relation between the
        endpoints of source and target is not totally defined.
        That is,. one of the point relations is None. By default this parameter
        is set to False.

    Returns
    -------
    result: Set[TLink]
        The union of the tlinks given as input and the tlinks inferred.
    """

    result = copy.deepcopy(tlinks)

    old_tlinks = tlinks
    new_tlinks = tlinks
    while True:

        inferred_tlinks = set()
        for tlink1 in old_tlinks:
            for tlink2 in new_tlinks:

                if tlink1 == tlink2:
                    continue

                inferred_tlink = tlink1 & tlink2

                if inferred_tlink and (inferred_tlink not in result):
                    inferred_tlinks.add(inferred_tlink)

        if inferred_tlinks:
            result.update(inferred_tlinks)
            old_tlinks = result
            new_tlinks = inferred_tlinks

        else:
            break

    return result
