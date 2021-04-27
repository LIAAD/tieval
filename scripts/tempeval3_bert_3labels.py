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

from text2timeline.datasets import tempeval3
from text2timeline import narrative as t2s
from text2timeline import temporal_closure as tc

import collections

import json

from pprint import pprint

from tqdm import tqdm

from typing import NamedTuple

import gc

import copy

from text2timeline.toolbox import BuildIO
from text2timeline.toolbox import ModelConfig
from text2timeline.toolbox import Model

from text2timeline.toolbox import print_header
from text2timeline.toolbox import print_metrics


# Fix GPU memory problem.
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)


"""Read datasets."""
TEMPEVAL_PATH = r'datasets/TempEval-3'
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


"""Preprocess datasets."""

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
