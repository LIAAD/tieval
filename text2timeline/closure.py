from typing import Set

from text2timeline.links import TLink
import copy


def temporal_closure(tlinks: Set[TLink]) -> Set[TLink]:

    n_tlinks = len(tlinks)
    while True:

        # find new tlinks
        inferred_tlinks = set()
        foo = copy.copy(tlinks)
        while foo:

            tlink = foo.pop()
            for tl in foo:
                inferred = tl & tlink

                if inferred:
                    inferred_tlinks.add(inferred)

        # update tlinks set
        tlinks.update(inferred_tlinks)

        # check if new tlinks were found
        new_n_tlinks = len(tlinks)
        if new_n_tlinks != n_tlinks:
            n_tlinks = new_n_tlinks

        else:
            break

    return tlinks
