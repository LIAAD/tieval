
class TemporalClosure:
    pass


"""
    def temporal_closure(self, tlinks) -> Dict:
        # TODO: Keep the original labels when the inferred are more ambiguous.
        # Remove duplicate temporal links.
        lids = set()
        relations = set()
        for lid, tlink in tlinks.items():
            scr = tlink.source
            tgt = tlink.target
            if (scr, tgt) not in relations and (tgt, scr) not in relations:
                lids.add(lid)
                relations.add((scr, tgt))

        tlinks = {lid: tlink for lid, tlink in tlinks.items() if lid in lids}

        # convert interval relations to point relations
        new_relations = {((tlink.source, scr_ep), r, (tlink.target, tgt_ep))
                         for lid, tlink in tlinks.items()
                         for scr_ep, r, tgt_ep in tlink.point_relation}

        # Add the symmetric of each relation. A < B ---> B > A
        {(tgt, _INVERSE_POINT_RELATION[rel], scr) for scr, rel, tgt in new_relations if rel != '='}

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
                    relation13 = _POINT_TRANSITIONS[relation12][relation23]
                    if not relation13:
                        continue
                    new_relation = (point1, relation13, point3)
                    if new_relation not in point_relations:
                        new_relations.add(new_relation)

        # Combine point relations.
        combined_point_relations = collections.defaultdict(list)
        for ((scr, scr_ep), rel, (tgt, tgt_ep)) in point_relations:
            if scr == tgt:
                continue
            combined_point_relations[(scr, tgt)] += [(scr_ep, rel, tgt_ep)]

        # Creat and store the valid temporal links.
        tlinks = []
        for (scr, tgt), point_relation in combined_point_relations.items():
            tlink = TLink(scr, tgt, point_relation=point_relation)
            if tlink.relation:
                tlinks.append(tlink)

        # Generate indexes for tlinks.
        return {f'lc{idx}': tlink for idx, tlink in enumerate(tlinks)}

"""