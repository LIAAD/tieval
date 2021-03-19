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

import text2story.data.tempeval3
from text2story import read_xml as rxml
from text2story import temporal_closure as tc
from text2story import utils

import collections

import json

from pprint import pprint

pd.set_option('display.max_columns', 40)

tokenizer = nltk.word_tokenize

"""Read data."""
TEMPEVAL_PATH = r'data/TempEval-3'
data = text2story.data.tempeval3.load_data(TEMPEVAL_PATH, tokenizer)
base = pd.concat([data['train'][dataset]['base'] for dataset in data['train']], axis=0)
tlinks = pd.concat([data['train'][dataset]['tlinks'] for dataset in data['train']], axis=0)

# Read test data.
base_test = data['test']['platinum']['base']
tlinks_test = data['test']['platinum']['tlinks']

del data

# Remove relations that mean the same thing.
tlinks['relType'] = tlinks.relType.map(tc.relevant_relations)
tlinks.drop_duplicates(inplace=True)
tlinks_test['relType'] = tlinks_test.relType.map(tc.relevant_relations)
tlinks_test.drop_duplicates(inplace=True)


"""Preprocess data."""
# Add text for each token and the context between them to tlinks_closure dataframe.
tlinks = utils.add_tokens(tlinks, base)
tlinks_test = utils.add_tokens(tlinks_test, base_test)

# Limit the temporal links to be inside the same sentence, or consecutive sentences.
sent_distance = tlinks.related_sent - tlinks.source_sent
tlinks = tlinks[(tlinks.source == 't0') | (tlinks.relatedTo == 't0') | (abs(sent_distance) <= 1)]

# Add context.
tlinks = utils.add_context(tlinks, base)
tlinks_test = utils.add_context(tlinks_test, base_test)

# Replace dct tokens by "<dct>".
DCT_TOKEN = '<dct>'
tlinks.loc[tlinks.source == 't0', 'source_text'] = DCT_TOKEN
tlinks.loc[tlinks.relatedTo == 't0', 'related_text'] = DCT_TOKEN

tlinks_test.loc[tlinks_test.source == 't0', 'source_text'] = DCT_TOKEN
tlinks_test.loc[tlinks_test.relatedTo == 't0', 'related_text'] = DCT_TOKEN

# Add point relation.
_start = 0
_end = 1
_te1 = 0  # Time expression 1
_te2 = 1  # Time expression 2

# Remove relations that mean the same thing.
_interval_to_point = {
    "BEFORE": [(_te1, _start, "<", _te2, _start),
               (_te1, _start, "<", _te2, _end),
               (_te1, _end, "<", _te2, _start),
               (_te1, _end, "<", _te2, _end)],
    "AFTER": [(_te1, _start, ">", _te2, _start),
              (_te1, _start, ">", _te2, _end),
              (_te1, _end, ">", _te2, _start),
              (_te1, _end, ">", _te2, _end)],
    "IBEFORE": [(_te1, _start, "<", _te2, _start),
                (_te1, _start, "=", _te2, _end),
                (_te1, _end, "<", _te2, _start),
                (_te1, _end, "<", _te2, _end)],
    "IAFTER": [(_te1, _start, ">", _te2, _start),
               (_te1, _start, "=", _te2, _end),
               (_te1, _end, ">", _te2, _start),
               (_te1, _end, ">", _te2, _end)],
    "CONTAINS": [(_te1, _start, "<", _te2, _start),
                 (_te1, _start, "<", _te2, _end),
                 (_te1, _end, ">", _te2, _start),
                 (_te1, _end, ">", _te2, _end)],
    "INCLUDES": [(_te1, _start, "<", _te2, _start),
                 (_te1, _start, "<", _te2, _end),
                 (_te1, _end, ">", _te2, _start),
                 (_te1, _end, ">", _te2, _end)],
    "IS_INCLUDED": [(_te1, _start, ">", _te2, _start),
                    (_te1, _start, "<", _te2, _end),
                    (_te1, _end, ">", _te2, _start),
                    (_te1, _end, "<", _te2, _end)],
    "BEGINS-ON": [(_te1, _start, "=", _te2, _start),
                  (_te1, _start, "<", _te2, _end),
                  (_te1, _end, ">", _te2, _start),
                  (_te1, _end, None, _te2, _end)],
    "ENDS-ON": [(_te1, _start, None, _te2, _start),
                (_te1, _start, "<", _te2, _end),
                (_te1, _end, ">", _te2, _start),
                (_te1, _end, "=", _te2, _end)],
    "BEGINS": [(_te1, _start, "=", _te2, _start),
               (_te1, _start, "<", _te2, _end),
               (_te1, _end, ">", _te2, _start),
               (_te1, _end, "<", _te2, _end)],
    "BEGUN_BY": [(_te1, _start, "=", _te2, _start),
                 (_te1, _start, "<", _te2, _end),
                 (_te1, _end, ">", _te2, _start),
                 (_te1, _end, ">", _te2, _end)],
    "ENDS": [(_te1, _start, ">", _te2, _start),
             (_te1, _start, "<", _te2, _end),
             (_te1, _end, ">", _te2, _start),
             (_te1, _end, "=", _te2, _end)],
    "ENDED_BY": [(_te1, _start, "<", _te2, _start),
                 (_te1, _start, "<", _te2, _end),
                 (_te1, _end, ">", _te2, _start),
                 (_te1, _end, "=", _te2, _end)],

    "SIMULTANEOUS": [(_te1, _start, "=", _te2, _start),
                     (_te1, _start, "<", _te2, _end),
                     (_te1, _end, ">", _te2, _start),
                     (_te1, _end, "=", _te2, _end)],
    "IDENTITY": [(_te1, _start, "=", _te2, _start),
                 (_te1, _start, "<", _te2, _end),
                 (_te1, _end, ">", _te2, _start),
                 (_te1, _end, "=", _te2, _end)],
    "DURING": [(_te1, _start, "=", _te2, _start),
               (_te1, _start, "<", _te2, _end),
               (_te1, _end, ">", _te2, _start),
               (_te1, _end, "=", _te2, _end)],
    "DURING_INV": [(_te1, _start, "=", _te2, _start),
                   (_te1, _start, "<", _te2, _end),
                   (_te1, _end, ">", _te2, _start),
                   (_te1, _end, "=", _te2, _end)],
    "OVERLAP": [(_te1, _start, "<", _te2, _start),
                (_te1, _start, "<", _te2, _end),
                (_te1, _end, ">", _te2, _start),
                (_te1, _end, "<", _te2, _end)]
}

map_relations = [[k, p1, p2, r] for k, v in _interval_to_point.items() for _, p1, r, _, p2 in v if r is not None]
map_relations = pd.DataFrame(map_relations, columns=['relType', 'edge1', 'edge2', 'pointRel'])
tlinks = tlinks.merge(map_relations)
tlinks_test = tlinks_test.merge(map_relations)

# Build inputs for the model.
X = tlinks.source_text + ' ' + tlinks.related_text + ' ' + tlinks.context
X_context = tlinks.context
X_events = tlinks.source_text + ' ' + tlinks.related_text
X_edge1 = tlinks.edge1.values
X_edge2 = tlinks.edge1.values

X_test = tlinks_test.source_text + ' ' + tlinks_test.related_text
X_context_test = tlinks_test.context
X_edge1_test = tlinks_test.edge1.values
X_edge2_test = tlinks_test.edge2.values

# Build target.
classes = tlinks.pointRel.unique()
n_classes = len(classes)

classes2idx = {cl: i for i, cl in enumerate(classes)}
idx2classes = dict((i, cl) for cl, i in classes2idx.items())

y = [classes2idx[cl] for cl in tlinks.pointRel]
y_test = [classes2idx[cl] for cl in tlinks_test.pointRel]

# Compute class weight.
class_count = collections.Counter(y)
n_samples = len(y)
class_weight = {cl: (n_samples / (n_classes * count)) for cl, count in class_count.items()}

# Split data into train and validation and build a tensorflow dataset.
data_size = len(tlinks)
batch_size = 32

cut = round(data_size * 0.8)
train_set = tf.data.Dataset.from_tensor_slices(
    ((X_context[:cut], X_events[:cut], X_edge1[:cut], X_edge2[:cut]), y[:cut])).batch(batch_size).prefetch(1)
valid_set = tf.data.Dataset.from_tensor_slices(
    ((X_context[cut:], X_events[cut:], X_edge1[cut:], X_edge2[cut:]), y[cut:])).batch(batch_size).prefetch(1)


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
        #class_weight=class_weight
    )

    Y_prob_valid = model.predict(valid_set)
    y_pred_valid = np.argmax(Y_prob_valid, axis=1)
    cm = confusion_matrix(y_valid, y_pred_valid)
    with open('results/results.txt', 'a') as f:
        f.write(model_name)
        f.write('\n')
        f.write(str(cm))
        f.write('\n\n')





# model = models.load_model(model_path)
model.load_weights(model_path)


"""Evaluate Model."""
# Validation set.
utils.print_confusion_matrix(y_valid, y_pred_valid, classes2idx.keys())

baseline_classe = np.argmax(np.bincount(y))

# Test set.
Y_proba_test = model.predict([X_test, X_edge1_test, X_edge2_test])
y_pred_test = np.argmax(Y_proba_test, axis=1)
print(confusion_matrix(y_test, y_pred_test))

# Compute temporal-awareness.
tlinks_test['relPredicted'] = [idx2classes[idx] for idx in y_pred_test]
tlinks_test['baseline'] = idx2classes[baseline_classe]

allen_rel2point_rel = {k: tuple([r for _, _, r, _, _ in v]) for k, v in _interval_to_point.items()}
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
