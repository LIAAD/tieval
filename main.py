import os

import nltk
import numpy as np
import pandas as pd

import tensorflow_hub as hub
import tensorflow_text as text  # Dependency for BERT preprocessing.

from official.nlp import optimization

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

import tensorflow as tf
from tensorflow.keras import callbacks
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import models
from tensorflow.keras import metrics

from text2story.data import tempeval3
from text2story import narrative as t2s
from text2story import temporal_closure as tc
from text2story import utils

import collections

import json

from pprint import pprint

from tqdm import tqdm

from typing import NamedTuple

import gc

import copy

import copy


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
            for i_rel, p_rel in t2s._INTERVAL_TO_POINT.items():
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


class BuildIO:
    def __init__(self, classes):
        """Build input output from the documents.

        :param classes: The classes in the target variable. In the point classification problem the classes would be
        ['<', '>', '=']. In the interval relation would be ['BEFORE', 'AFTER', (...)].
        """
        self.target = Target(classes)
        self.inputs = dict()
        self.outputs = dict()


    def run(self, docs):
        """

        :param docs: The
        :return:
        """
        X_context = []
        X_event = []
        X_edge1 = []
        X_edge2 = []
        relations = []

        for doc in docs:
            expressions = self.get_expressions(doc)
            for tlink in doc.tlinks:
                # get the source and target expressions.
                scr_exp = expressions[tlink.source]
                tgt_exp = expressions[tlink.target]

                point_rels = tlink.complete_point_relation()
                for point_rel in point_rels:
                    edge1, relation, edge2 = point_rel
                    event = self.get_text(tlink, scr_exp, tgt_exp)
                    context = self.get_context(doc, tlink, scr_exp, tgt_exp)

                    X_context.append(context)
                    X_event.append(event)
                    X_edge1.append(edge1)
                    X_edge2.append(edge2)
                    relations.append(relation)

        y = self.target.indexes(relations)
        return (X_context, X_event, X_edge1, X_edge2), y

    def run_single_tlink(self, tlink, doc):
        """

        :param docs: The
        :return:
        """
        X_context = []
        X_event = []
        X_edge1 = []
        X_edge2 = []
        relations = []

        expressions = self.get_expressions(doc)

        # get the source and target expressions.
        scr_exp = expressions[tlink.source]
        tgt_exp = expressions[tlink.target]

        point_rels = tlink.complete_point_relation()
        for point_rel in point_rels:
            edge1, relation, edge2 = point_rel
            event = self.get_text(tlink, scr_exp, tgt_exp)
            context = self.get_context(doc, tlink, scr_exp, tgt_exp)

            X_context.append(context)
            X_event.append(event)
            X_edge1.append(edge1)
            X_edge2.append(edge2)
            relations.append(relation)

        y = self.target.indexes(relations)
        return (X_context, X_event, X_edge1, X_edge2), y

    def build_tensorflow_dataset(self, docs, batch_size=32, oversample=False, name: str=None):
        X, y = self.run(docs)

        if name:
            self.inputs[name] = X
            self.outputs[name] = y

        if oversample:
            X, y = self.oversample(X, y)

        dataset = tf.data.Dataset.from_tensor_slices((X, y)).batch(batch_size).prefetch(1)
        return dataset

    @staticmethod
    def get_text(tlink, source, target):
        # get text of each expression.
        # replace the DCT text by the '<dct>' token.
        if tlink.source == 't0':
            return ' '.join(['<dct>', target.text])
        elif tlink.target == 't0':
            return ' '.join([source.text, '<dct>'])
        else:
            return ' '.join([source.text, target.text])

    @staticmethod
    def get_context(doc, tlink, source, target):
        # get the context between the two expressions.
        # if the source or target are the DCT ('t0') we use as context the sentence where the other term
        # appeared. else the context is the tokens between each of the expressions.
        if tlink.source == 't0':
            start_context, end_context = target.endpoints
            return ' '.join(sent for s, e, sent in doc.sentences if start_context >= s and end_context <= e)
        elif tlink.target == 't0':
            start_context, end_context = source.endpoints
            return ' '.join(sent for s, e, sent in doc.sentences if start_context >= s and end_context <= e)
        else:
            start_context = min(source.endpoints[0], target.endpoints[0])
            end_context = max(target.endpoints[1], target.endpoints[1])
            return ' '.join(token for s, e, token in doc.tokens if start_context <= s and end_context >= e)

    @staticmethod
    def get_expressions(doc):
        expressions = {eiid: event for event in doc.events if hasattr(event, 'eiid') for eiid in event.eiid}
        expressions.update({timex.tid: timex for timex in doc.timexs})
        return expressions

    def oversample(self, X: list, y: list, verbose: bool = True):
        oversample_idxs = np.array([], dtype=int)
        class_count = collections.Counter(y)

        if verbose:
            print("Original class count:")
            self.print_class_count(class_count)

        max_class_count = max(class_count.values())
        for class_, count in class_count.items():
            class_idxs = np.where(y == class_)[0]
            if count < max_class_count:
                sample_idxs = class_idxs[np.random.randint(0, len(class_idxs), max_class_count)]
            else:
                sample_idxs = class_idxs
            oversample_idxs = np.append(oversample_idxs, sample_idxs)

        np.random.shuffle(oversample_idxs)

        X_oversample = tuple([[x[idx] for idx in oversample_idxs] for x in X])
        y_oversample = list(y[oversample_idxs])

        if verbose:
            print("Oversample class count:")
            self.print_class_count(collections.Counter(y_oversample))

        return X_oversample, y_oversample

    def print_class_count(self, counter: dict):
        for class_, count in counter.items():
            print(f"\t{self.target.idx2classes[class_]}: {count}")


class ModelConfig(NamedTuple):
    """ Model configuration.

    name: Model name.
    handler_url: Url to BERT handler.
    bert_url: Url to BERT model.
    epochs: Number of epochs to train
    output_size: Output size.
    train_set: Train set.
    valid_set: Validation set.
    init_lr: Initial learning rate.
    """
    name: str
    handler_url: str
    bert_url: str
    output_size: int
    epochs: int = 5
    init_lr: float = 3e-5


class Model:
    def __init__(self, config: ModelConfig):
        self.name = config.name
        self.path = f'models/{model_name}_based_model_context_point_relation'

        self.handler_url = config.handler_url
        self.bert_url = config.bert_url

        self.epochs = config.epochs
        self.output_size = config.output_size
        self.init_lr = config.init_lr

        self.model = None

    def build(self):
        context = layers.Input(shape=(), dtype=tf.string, name='context')
        events = layers.Input(shape=(), dtype=tf.string, name='source')

        # Tokenize the text to word pieces.
        bert_preprocess = hub.load(self.handler_url)
        tokenizer = hub.KerasLayer(bert_preprocess.tokenize, name='tokenizer')
        segments = [tokenizer(s) for s in [context, events]]
        packer = hub.KerasLayer(bert_preprocess.bert_pack_inputs,
                                arguments=dict(seq_length=256),
                                name='packer')
        encoder_input = packer(segments)

        edge1 = layers.Input(shape=(1), dtype=tf.float32, name='edge1')
        edge2 = layers.Input(shape=(1), dtype=tf.float32, name='edge2')
        encoder = hub.KerasLayer(self.bert_url, trainable=True)

        z = encoder(encoder_input)['pooled_output']
        z = layers.concatenate([z, edge1, edge2])
        z = layers.Dense(256, activation='relu')(z)
        z = layers.Dense(128, activation='relu')(z)
        z = layers.Dense(64, activation='relu')(z)
        z = layers.Dense(32, activation='relu')(z)
        output = layers.Dense(self.output_size)(z)

        self.model = models.Model([context, events, edge1, edge2], output)

    def compile(self, train_set):
        steps_per_epoch = tf.data.experimental.cardinality(train_set).numpy()
        num_train_steps = self.epochs * steps_per_epoch
        num_warmup_steps = int(0.1 * num_train_steps)
        optimizer = optimization.create_optimizer(
            init_lr=3e-5,
            num_train_steps=num_train_steps,
            num_warmup_steps=num_warmup_steps,
            optimizer_type='adamw'
        )

        self.model.compile(
            optimizer=optimizer,
            loss=losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

    def fit(self, train_set, valid_set):
        checkpoint_cb = callbacks.ModelCheckpoint(self.path, save_best_only=True, save_weights_only=True)
        early_stop_cb = callbacks.EarlyStopping(patience=2, verbose=1, restore_best_weights=True)

        self.model.fit(
            train_set,
            validation_data=valid_set,
            epochs=self.epochs,
            callbacks=[early_stop_cb, checkpoint_cb],
        )

    def train(self, train_set, valid_set):
        self.build()
        self.compile(train_set)
        self.fit(train_set, valid_set)

    def load(self):
        self.build()
        self.model.load_weights(self.path)

    def predict(self, dataset):
        return self.model.predict(dataset)

    def predict_classes(self, docs, builder):
        pred_classes = []
        for doc in tqdm(docs):
            doc_set = builder.build_tensorflow_dataset(docs=[doc], batch_size=4)
            for X_tlk, y_tlk in doc_set:
                Y_tlk_pred = self.model.predict(X_tlk)
                relation = builder.target.prediction_to_relation(Y_tlk_pred)
                pred_classes.append(relation)
        return pred_classes


def print_header(msg):
    print('-' * len(msg))
    print(msg)
    print('-' * len(msg))


def print_metrics(y_true, y_pred):
    metrics_dict = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='macro'),
        'Recall': recall_score(y_true, y_pred, average='macro'),
    }
    for metric, value in metrics_dict.items():
        print(f'\t{metric}: {value:.03}')

    cm = confusion_matrix(y_true, y_pred)
    print(f'\tConfusion Matrix:')
    for line in cm:
        print(f'\t\t{line}')


# Fix GPU memory problem.
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)


"""Read data."""
TEMPEVAL_PATH = r'data/TempEval-3'
data = tempeval3.load_data(TEMPEVAL_PATH)

train_valid_docs = data['train']['aquaint'] + data['train']['timebank']
cut = int(len(train_valid_docs) * 0.85)
train_docs = train_valid_docs[:cut]
valid_docs = train_valid_docs[cut:]
test_docs = data['test']['platinum']

print_header('Data size (documents)')
print(f"Number of documents in train: {len(train_docs)}")
print(f"Number of documents in validation: {len(valid_docs)}")
print(f"Number of documents in test: {len(test_docs)}")
print()


"""Preprocess data."""

# Augment relations in the train set.
augm_docs = copy.deepcopy(train_docs)
for doc in augm_docs:
    doc.augment_tlinks('=')

# Limit to tasks A, B and C. That is, remove the relations with document creation time.

# Build train_valid_set.
batch_size = 32
classes = ['>', '<', '=']
builder = BuildIO(classes)
train_set = builder.build_tensorflow_dataset(augm_docs, batch_size, name='train')
valid_set = builder.build_tensorflow_dataset(valid_docs, batch_size, name='valid')
test_set = builder.build_tensorflow_dataset(test_docs, batch_size, name='test')

print_header('Data size (tlinks)')
print(f"Number of tlinks in train: {len(train_set) * batch_size}")
print(f"Number of tlinks in validation: {len(valid_set) * batch_size}")
print(f"Number of tlinks in test : {len(test_set) * batch_size}")


# Load BERT urls.
with open('resources/bert_urls.json', 'r') as f:
    bert_urls = json.load(f)

# Keep just the small BERTs (the others can't be loaded in the GPU memory).
small_bert_urls = {name.replace('/', '_'): bert_urls[name] for name in bert_urls if name.startswith('small_bert')}


"""Train model."""
# Evaluate different architectures.
y_valid = np.concatenate([y for x, y in valid_set])
for model_name in tqdm(small_bert_urls):
    # model_name = 'small_bert_bert_en_uncased_L-8_H-512_A-8'
    config = ModelConfig(
        name=model_name,
        handler_url=small_bert_urls[model_name]['handler'],
        bert_url=small_bert_urls[model_name]['model'],
        output_size=len(classes)
    )

    try:
        model = Model(config)
        model.train()
    except:
        message = f"Could not train model {model_name}"
        print()
        print(message)
        print()

        with open('logs/log.txt', 'a') as f:
            f.write(message)
            f.write('\n')
            gc.collect()
        continue

    Y_prob_valid = model.predict(valid_set)
    y_pred_valid = np.argmax(Y_prob_valid, axis=1)
    cm = confusion_matrix(y_valid, y_pred_valid)
    with open('results/results.txt', 'a') as f:
        f.write(model_name)
        f.write('\n')
        f.write(str(cm))
        f.write('\n\n')

    with open('logs/log.txt', 'a') as f:
        f.write(model_name + ' done')
        f.write('\n')

    gc.collect()


""" Evaluate model."""
best_models = [
    'small_bert_bert_en_uncased_L-8_H-512_A-8',
    'small_bert_bert_en_uncased_L-10_H-512_A-8',
    'small_bert_bert_en_uncased_L-12_H-256_A-4',
]

model_name = best_models[0]
config = ModelConfig(
    name=model_name,
    handler_url=small_bert_urls[model_name]['handler'],
    bert_url=small_bert_urls[model_name]['model'],
    output_size=len(classes)
)

model = Model(config)
model.load()


# Validation set.
y_valid = builder.outputs['valid']
Y_prob_valid = model.predict(valid_set)
y_pred_valid = np.argmax(Y_prob_valid, axis=1)

print_header('Validation')
print_metrics(y_valid, y_pred_valid)


# Test set.
y_test = builder.outputs['test']
Y_proba_test = model.predict(test_set)
y_pred_test = np.argmax(Y_proba_test, axis=1)

print_header('Test')
print_metrics(y_test, y_pred_test)


"""Evaluate with interval relations."""

rel_valid_pred = model.predict_classes(valid_docs, builder)
rel_valid = [tlink.interval_relation for doc in valid_docs for tlink in doc.tlinks]
print_header('Validation')
print("Accuracy:", accuracy_score(rel_valid, rel_valid_pred))
print(confusion_matrix(rel_valid, rel_valid_pred))


rel_test_pred = model.predict_classes(test_docs, builder)
rel_test = [tlink.interval_relation for doc in test_docs for tlink in doc.tlinks]
print_header('Test')
print("Accuracy:", accuracy_score(rel_test, rel_test_pred))
print(confusion_matrix(rel_test, rel_test_pred))

