from typing import Set

from text2timeline.links import TLink
import copy


def temporal_closure(tlinks: Set[TLink]) -> Set[TLink]:

    n_tlinks = len(tlinks)
    new_tlinks = copy.copy(tlinks)
    while new_tlinks:

        tlinks.update(new_tlinks)

        # find new tlinks
        tlink = tlinks.pop()
        infered_tlinks = [tl & tlink for tl in tlinks]

        for tl in tlinks:
            print(tl)
            print(tlink)
            infered = tl & tlink
            print(infered)
            print()







