from typing import Set

from text2timeline.links import TLink
import copy


def temporal_closure(
        tlinks: Set[TLink],
        allow_incomplete: bool = False) -> Set[TLink]:
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

    n_tlinks = len(tlinks)
    result = copy.copy(tlinks)
    while True:

        # find new tlinks
        inferred_tlinks = set()
        foo = copy.copy(result)
        while foo:

            tlink = foo.pop()
            for tl in foo:

                inferred = tl & tlink

                if inferred:

                    if not allow_incomplete:
                        if None in inferred.relation.point.relation:
                            continue

                    inferred_tlinks.add(inferred)

        # update tlinks set
        result.update(inferred_tlinks)

        # check if new tlinks were found
        new_n_tlinks = len(result)
        if new_n_tlinks != n_tlinks:
            n_tlinks = new_n_tlinks

        else:
            break

    return result
