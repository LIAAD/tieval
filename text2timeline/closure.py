from typing import Set

from text2timeline.links import TLink
import copy


def temporal_closure(tlinks: Set[TLink]) -> Set[TLink]:

    inferred_tlinks = []
    for tlink0 in tlinks:
        for tlink1 in tlinks:
            inferred = tlink0 & tlink1
            if inferred:
                print(tlink0, tlink1, inferred)
                inferred_tlinks += [inferred]
                inferred = tlink0 & tlink1

    n_tlinks = len(tlinks)
    new_tlinks = copy.copy(tlinks)
    while tlinks:

        # find new tlinks
        tlink = tlinks.pop()

        for tl in tlinks:
            print(tlink, tl, tl & tlink)
            r = tl & tlink

        inferred_tlinks = [tl & tlink for tl in tlinks if tl & tlink]
        new_tlinks.update(inferred_tlinks)

        new_n_tlinks = len(new_tlinks)

        if new_n_tlinks > n_tlinks:
            tlinks = copy.copy(new_tlinks)
