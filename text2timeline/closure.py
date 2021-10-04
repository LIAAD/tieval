from typing import Set

from text2timeline.links import TLink
import copy


def temporal_closure(tlinks: Set[TLink], allow_incomplete=False) -> Set[TLink]:

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
