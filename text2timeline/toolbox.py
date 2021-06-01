import numpy as np

import text2timeline.base as t2t


class Target:
    def __init__(self, classes):
        self.classes = classes
        self.n_classes = len(classes)

        self.classes2idx = {cl: i for i, cl in enumerate(classes)}
        self.idx2classes = dict((i, cl) for cl, i in self.classes2idx.items())

    def classes(self, indexes):
        return np.array([self.idx2classes[idx] for idx in indexes])

    def indexes(self, classes):
        return np.array([self.classes2idx[cl] for cl in classes])

    def prediction_to_relation(self, Y_pred):
        """Predicts the interval relation given the model prediction for the point relations.

        :param Y_pred:
        :return:
        """
        y_pred = np.argmax(Y_pred, axis=1)
        confidance = np.max(Y_pred, axis=1)

        pred_point_relations = self.format_relation(y_pred)
        cand_rels, cand_conf = self.generate_candidate_relations(confidance, pred_point_relations)
        ordered_relations = [rel for _, rel in sorted(zip(cand_conf, cand_rels))]
        relation = self.find_interval_relation(ordered_relations)
        return relation

    @staticmethod
    def find_interval_relation(point_relations):
        """Finds the interval relation for the poin relations.
        :param point_relations:
        :return: If the inputted point relation is valid it will return the corresponding interval relations. If not,
        it will return None.
        """
        for rel in point_relations:
            for i_rel, p_rel in t2t._INTERVAL_TO_POINT.items():
                if rel == p_rel:
                    return i_rel

    def format_relation(self, y_pred):
        """

        :param y_pred: the prediction made for each of the 4 endpoints.
        :return: the formated relation between those endpoints.
        :example:
            >>> prediction_to_relation([0, 2, 1, 0])
            [(0, '>', 0), (0, '=', 1), (1, '<', 0), (1, '>', 1)]

        """
        predicted_relation = [self.idx2classes[idx] for idx in y_pred]
        relations = [
            (0, predicted_relation[0], 0),
            (0, predicted_relation[1], 1),
            (1, predicted_relation[2], 0),
            (1, predicted_relation[3], 1)
        ]
        return relations

    @staticmethod
    def generate_candidate_relations(confidence, relations):
        """Generates the candidates based on the confidence that the model had on each of them.
        Since each of the interval relations can be resumed to one or two point relation we use this simple method to
        generate the candidates.
        This method outputs the single relations and the pairwise relations.
        The confidence is computed as follow:
            - single relation: the input confidence
            - pairwise relations: mean confidence of the relations.

        :param confidence: a list with the confidence of model prediction.
        :param relations: a list with the formated point realtions.
        :return: a tuple with the candidate relations and the confidence computed for that relation.

        """
        n = len(confidence)
        conf = []
        rels = []
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    conf.append(confidence[i])
                    rels.append([relations[i]])
                else:
                    conf.append((confidence[i] + confidence[j]) / 2)
                    rels.append([relations[i], relations[j]])
        return rels, conf

    def one_hot(self, indexes):
        oh = np.zeros([len(indexes), self.n_classes])
        oh[np.arange(len(indexes)), indexes] = 1
        return oh
