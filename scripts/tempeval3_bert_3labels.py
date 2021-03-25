import nltk
import numpy as np
import pandas as pd

import tensorflow_hub as hub
import tensorflow_text as text  # Dependency for BERT preprocessing.

from official.nlp import optimization

from sklearn.metrics import confusion_matrix

import tensorflow as tf
from tensorflow.keras import callbacks
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import models
from tensorflow.keras import metrics

from text2story.data import tempeval3
from text2story import temporal_closure as tc
from text2story import utils

import collections

import json

from pprint import pprint


"""Read data."""
TEMPEVAL_PATH = r'data/TempEval-3'
data = tempeval3.load_data(TEMPEVAL_PATH)

train_docs = data['train']['aquaint'] + data['train']['timebank']
test_docs = data['test']['platinum']


"""Preprocess data."""
# Replace dct tokens by "<dct>".
for doc in train_docs:
    expressions = {event.eiid: event for event in doc.events if hasattr(event, 'eiid')}
    expressions.update({timex.tid: timex for timex in doc.timexs})
    for tlink in doc.tlinks:
        # get the source and target expressions.
        scr_exp = expressions[tlink.source]
        tgt_exp = expressions[tlink.target]

        point_tlinks = tlink.complete_point_relation()

        # get the context between the two expressions.
        scr_exp.text
        tgt_exp.text
        start_context = scr_exp.endpoints[0]
        end_context = tgt_exp.endpoints[1]
        doc.text[start_context: end_context]


# Build inputs for the model.
X_context = tlinks.context
X_events = tlinks.source_text + ' ' + tlinks.related_text
X_edge1 = tlinks.edge1.values
X_edge2 = tlinks.edge2.values
X = np.array(list(zip(X_context, X_events, X_edge1, X_edge2)))

X_context_test = tlinks_test.context
X_events_test = tlinks_test.source_text + ' ' + tlinks_test.related_text
X_edge1_test = tlinks_test.edge1.values
X_edge2_test = tlinks_test.edge2.values
X_test = np.array(list(zip(X_context_test, X_events_test, X_edge1_test, X_edge2_test)))

# Build target.
classes = tlinks.pointRel.unique()
n_classes = len(classes)

classes2idx = {cl: i for i, cl in enumerate(classes)}
idx2classes = dict((i, cl) for cl, i in classes2idx.items())

y = np.array([classes2idx[cl] for cl in tlinks.pointRel])
y_test = np.array([classes2idx[cl] for cl in tlinks_test.pointRel])


"""
cut = round(data_size * 0.8)
train_set = tf.data.Dataset.from_tensor_slices(
    ((X_context[:cut], X_events[:cut], X_edge1[:cut], X_edge2[:cut]), y[:cut])).batch(batch_size).prefetch(1)
valid_set = tf.data.Dataset.from_tensor_slices(
    ((X_context[cut:], X_events[cut:], X_edge1[cut:], X_edge2[cut:]), y[cut:])).batch(batch_size).prefetch(1)
"""

oversample_idxs = np.array([], dtype=int)
class_count = collections.Counter(y)
max_class_count = max(class_count.values())
for class_, count in class_count.items():
    class_idxs = np.where(y == class_)[0]
    if count < max_class_count:
        sample_idxs = class_idxs[np.random.randint(0, len(class_idxs), max_class_count)]
    else:
        sample_idxs = class_idxs
    oversample_idxs = np.append(oversample_idxs, sample_idxs)
np.random.shuffle(oversample_idxs)


#X_oversample, y_oversample = utils.oversample(X, y)
oversample_size = 30000
X_oversample, y_oversample = X[oversample_idxs[:oversample_size]], y[oversample_idxs[:oversample_size]]

# Split data into train and validation and build a tensorflow dataset.
data_size = len(tlinks)
batch_size = 32
cut = round(oversample_size * 0.8)

X_train, y_train = X_oversample[:cut], y_oversample[:cut]
X_valid, y_valid = X_oversample[cut:], y_oversample[cut:]

train_set = tf.data.Dataset.from_tensor_slices(
    ((X_train[:, 0], X_train[:, 1], np.array(X_train[:, 2], dtype=float), np.array(X_train[:, 3], dtype=float)), y_train)).batch(batch_size).prefetch(1)
valid_set = tf.data.Dataset.from_tensor_slices(
    ((X_valid[:, 0], X_valid[:, 1], np.array(X_valid[:, 2], dtype=float), np.array(X_valid[:, 3], dtype=float)), y_valid)).batch(batch_size).prefetch(1)


"""Model."""
with open('resources/bert_urls.json', 'r') as f:
    bert_urls = json.load(f)

small_bert_urls = {name.replace('/', '_'): bert_urls[name] for name in bert_urls if name.startswith('small_bert')}


def build_model(handler_url, bert_url):

    context = layers.Input(shape=(), dtype=tf.string, name='context')
    events = layers.Input(shape=(), dtype=tf.string, name='source')

    # Tokenize the text to word pieces.
    bert_preprocess = hub.load(handler_url)
    tokenizer = hub.KerasLayer(bert_preprocess.tokenize, name='tokenizer')
    segments = [tokenizer(s) for s in [context, events]]
    packer = hub.KerasLayer(bert_preprocess.bert_pack_inputs,
                            arguments=dict(seq_length=256),
                            name='packer')
    encoder_input = packer(segments)

    edge1 = layers.Input(shape=(1), dtype=tf.float32, name='edge1')
    edge2 = layers.Input(shape=(1), dtype=tf.float32, name='edge2')
    encoder = hub.KerasLayer(bert_url, trainable=True)
    z = encoder(encoder_input)['pooled_output']
    z = layers.concatenate([z, edge1, edge2])
    z = layers.Dense(256, activation='relu')(z)
    z = layers.Dense(128, activation='relu')(z)
    z = layers.Dense(64, activation='relu')(z)
    z = layers.Dense(32, activation='relu')(z)
    output = layers.Dense(n_classes)(z)
    return models.Model([context, events, edge1, edge2], output)


def compile(model, epochs=5):
    steps_per_epoch = tf.data.experimental.cardinality(train_set).numpy()  # np.lower(len(X) / batch_size)
    num_train_steps = epochs * steps_per_epoch
    num_warmup_steps = int(0.1 * num_train_steps)
    optimizer = optimization.create_optimizer(
        init_lr=3e-5,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        optimizer_type='adamw'
    )

    model.compile(
        optimizer=optimizer,
        loss=losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    return model


y_valid = np.concatenate([y for x, y in valid_set])


for model_name in small_bert_urls:
    # model_name = 'small_bert_bert_en_uncased_L-4_H-768_A-12'
    epochs = 5
    model = build_model(
        handler_url=small_bert_urls[model_name]['handler'],
        bert_url=small_bert_urls[model_name]['model']
    )
    model = compile(model, epochs=epochs)

    model_path = f'models/{model_name}_based_model_context_point_relation'
    checkpoint_cb = callbacks.ModelCheckpoint(model_path, save_best_only=True, save_weights_only=True)
    early_stop_cb = callbacks.EarlyStopping(patience=2, verbose=1, restore_best_weights=True)
    reduce_lr_cb = callbacks.ReduceLROnPlateau(patience=1, verbose=1, min_lr=1E-6)

    model.fit(
        train_set,
        validation_data=valid_set,
        epochs=epochs,
        callbacks=[early_stop_cb, checkpoint_cb],
    )

    Y_prob_valid = model.predict(valid_set)
    y_pred_valid = np.argmax(Y_prob_valid, axis=1)
    cm = confusion_matrix(y_valid, y_pred_valid)
    with open('results/results.txt', 'a') as f:
        f.write(model_name)
        f.write('\n')
        f.write(str(cm))
        f.write('\n\n')
    break


# model = models.load_model(model_path)
model.load_weights(model_path)


"""Evaluate Model."""
# Validation set.
y_valid = np.concatenate([y for x, y in valid_set])
Y_prob_valid = model.predict(valid_set)
y_pred_valid = np.argmax(Y_prob_valid, axis=1)
utils.print_confusion_matrix(y_valid, y_pred_valid, classes2idx.keys())

baseline_class = np.argmax(np.bincount(y))

# Test set.
Y_proba_test = model.predict([X_test[:, 0], X_test[:, 1], np.array(X_test[:, 2], dtype=float), np.array(X_test[:, 3], dtype=float)])
y_pred_test = np.argmax(Y_proba_test, axis=1)
print(confusion_matrix(y_test, y_pred_test))


# From point relation to interval relation.
tlinks_test['y_proba'] = np.max(Y_proba_test, axis=1)
tlinks_test['y_pred'] = y_pred_test

valid_relations = tlinks.relType.unique()

interval2point = {rel: tc._interval_to_point[rel] for rel in valid_relations}

def point2interval(point_relations):
    for n_relations in range(1, 5):
        relations = point_relations[:n_relations]
        for interval_rel, point_rel in interval2point.items():
            cond = [True if rel in relations else False for rel in point_rel]
            if all(cond):
                return interval_rel

pred_relation = []
point_pred = tlinks_test.groupby(['file', 'lid'])['y_proba', 'y_pred'].apply(lambda x: list(x.values)).to_dict()
formatted_point_pred = dict()
for link, values in point_pred.items():
    confidence = [conf for conf, _ in values]
    relations = [idx2classes[rel] for _, rel in values]
    formatted = [
        (0, 0, relations[0], 1, 0),
        (0, 0, relations[1], 1, 1),
        (0, 1, relations[2], 1, 0),
        (0, 1, relations[3], 1, 1)
    ]

    sorted_relations = [rel for _, rel in sorted(zip(confidence, formatted), reverse=True)]

    pred_relation.append(point2interval(sorted_relations))

true = [tlinks_test.relType[i] for i in range(0, len(tlinks_test.relType), 4)]
pred_relation

pprint(list(zip(true, pred_relation)))


example = [(0, 0, '=', 1, 0), (0, 1, '<', 1, 1), (0, 0, '<', 1, 1), (0, 1, '<', 1, 0)]  # BEGINS
example = [(0, 1, '<', 1, 0), (0, 0, '=', 1, 0), (0, 1, '<', 1, 1), (0, 0, '<', 1, 1)]
point2interval(example)

# Compute temporal-awareness.
tlinks_test['relPredicted'] = [idx2classes[idx] for idx in y_pred_test]
tlinks_test['baseline'] = idx2classes[baseline_class]

allen_rel2point_rel = {k: tuple([r for _, _, r, _, _ in v]) for k, v in utils.interval_to_point.items()}
point_rel2allen_rel = {v: k for k, v in allen_rel2point_rel.items()}

grouped_point_rel = tlinks_test.groupby(['file', 'lid', 'relType']).relPredicted.apply(tuple)

pred_allen_rel = []
for rel in grouped_point_rel:
    if rel in point_rel2allen_rel:
        pred_allen_rel.append(point_rel2allen_rel[rel])
    else:
        pred_allen_rel.append('')

annotations_test = tc.get_annotations(tlinks_test, ['source', 'relatedTo', 'pointRel'])
annotations_pred = tc.get_annotations(tlinks_test, ['source', 'relatedTo', 'relPredicted'])
annotations_baseline = tc.get_annotations(tlinks_test, ['source', 'relatedTo', 'baseline'])

ta = tc.multifile_temporal_awareness(annotations_test, annotations_pred)
ta_baseline = tc.multifile_temporal_awareness(annotations_test, annotations_baseline)

print(f"Temporal awareness baseline: {ta_baseline:.3}")
print(f"Temporal awareness model: {ta:.3}")
