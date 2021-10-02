from typing import Set

from text2timeline.links import TLink
import copy


def temporal_closure(tlinks: Set[TLink]) -> Set[TLink]:

    new_tlinks = copy.deepcopy(tlinks)

    while new_tlinks:

        tlinks.update(new_tlinks)

        # find new tlinks
        old_tlinks = copy.deepcopy(tlinks)
        while old_tlinks:
            link = old_tlinks.pop()

            inferred_links = [link & l for l in old_tlinks]






