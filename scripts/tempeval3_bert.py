import nltk
import numpy as np
import pandas as pd

import tensorflow_hub as hub
import tensorflow_text as text

from official.nlp import optimization

from sklearn.metrics import confusion_matrix

import tensorflow as tf
from tensorflow.keras import callbacks
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import models

from text2timeline import read_xml as rxml
from text2timeline import temporal_closure as tc
from text2timeline import utils

pd.set_option('display.max_columns', 40)

tokenizer = nltk.word_tokenize

"""Read datasets."""
# The training set will be the merge of TimeBank and AQUAINT.
AQUAINT_PATH = r'../data/TempEval-3/Train/TBAQ-cleaned/AQUAINT'
TIMEBANK_PATH = r'../data/TempEval-3/Train/TBAQ-cleaned/TimeBank'

base_aquaint, tlinks_aquaint = rxml.read_tempeval3(AQUAINT_PATH, tokenizer)
base_timebank, tlinks_timebank = rxml.read_tempeval3(TIMEBANK_PATH, tokenizer)

base = pd.concat([base_timebank, base_aquaint], axis=0)
tlinks = pd.concat([tlinks_timebank, tlinks_aquaint], axis=0)

# Read test datasets.
TEST_PATH = r'../data/TempEval-3/Test/TempEval-3-Platinum'
base_test, tlinks_test = rxml.read_tempeval3(TEST_PATH, tokenizer)

# Remove relations that mean the same thing.
tlinks_test['relType'] = tlinks_test.relType.map(tc.relevant_relations)
tlinks_test.drop_duplicates(inplace=True)

"""Preprocess datasets."""
# Remove relations that mean the same thing.
reduce_relations = {
    'OVERLAP': 'OVERLAP',
    'ENDS-ON': 'INCLUDES',
    'ENDS': 'INCLUDES',
    'BEGINS-ON': 'IS_INCLUDED',
    'BEFORE': 'BEFORE',
    'SIMULTANEOUS': 'INCLUDES',
    'AFTER': 'AFTER',
    'BEGINS': 'IS_INCLUDED',
    'IS_INCLUDED': 'IS_INCLUDED',
    'ENDED_BY': 'BEFORE',
    'INCLUDES': 'INCLUDES',
    'IBEFORE': 'BEFORE',
    'IAFTER': 'AFTER',
    'BEGUN_BY': 'AFTER',
    'CONTAINS': 'INCLUDES',
    'IDENTITY': 'SIMULTANEOUS',
    'DURING': 'SIMULTANEOUS',
    'DURING_INV': 'SIMULTANEOUS'
}

tlinks['relType'] = tlinks.relType.map(reduce_relations)
tlinks_test['relType'] = tlinks_test.relType.map(reduce_relations)

# Add text for each token and the context between them to tlinks_closure dataframe.
tlinks = utils.add_tokens(tlinks, base)
tlinks_test = utils.add_tokens(tlinks_test, base_test)

# Limit the temporal links to be inside the same sentece, or betwen neighbor sentences.
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

# Build inputs for the model.
X_source = tlinks.source_text.values
X_related = tlinks.related_text.values
X_context = tlinks.context.values
X = tlinks.source_text + ' ' + tlinks.related_text + ' ' + tlinks.context

# input_len = [len(tokenizer(x)) for x in tqdm(X)]

# Build target.
classes = tlinks.relType.unique()
output_size = len(classes)

classes2idx = {cl: i for i, cl in enumerate(classes)}
idx2classes = dict((i, cl) for cl, i in classes2idx.items())

y = [classes2idx[cl] for cl in tlinks.relType]

# Split datasets into train and validation and build a tensorflow train_valid_set.
data_size = len(tlinks)
batch_size = 64

cut = round(data_size * 0.8)
train_set = tf.data.Dataset.from_tensor_slices(
    (X[:cut], y[:cut])).batch(batch_size).prefetch(1)
valid_set = tf.data.Dataset.from_tensor_slices(
    (X[cut:], y[cut:])).batch(batch_size).prefetch(1)

"""Model."""
preprocess_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'
bert_url = 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-256_A-4/1'


def build_model():
    input = layers.Input(shape=(), dtype=tf.string, name='text1')
    preprocess_layer = hub.KerasLayer(preprocess_url, name='preprocessing')
    encoder_input = preprocess_layer(input)
    encoder = hub.KerasLayer(bert_url, trainable=True)
    z = encoder(encoder_input)
    z = layers.Dense(256, activation='relu')(z['pooled_output'])
    z = layers.Dense(128, activation='relu')(z)
    z = layers.Dense(64, activation='relu')(z)
    z = layers.Dense(32, activation='relu')(z)
    output = layers.Dense(output_size)(z)
    return models.Model(input, output)


model = build_model()

epochs = 5
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

model_path = r'bert_L-4_H-256_A-4_based_model_context_reduce_labels_LARGER'
checkpoint_cb = callbacks.ModelCheckpoint(model_path, save_best_only=True, save_weights_only=True)
early_stop_cb = callbacks.EarlyStopping(patience=2, verbose=1, restore_best_weights=True)
reduce_lr_cb = callbacks.ReduceLROnPlateau(patience=1, verbose=1, min_lr=1E-6)

model.fit(train_set, validation_data=valid_set, epochs=epochs, callbacks=[early_stop_cb, checkpoint_cb])

# model = models.load_model(model_path)

"""Evaluate Model."""
# Validation set.
y_valid = np.concatenate([y for x, y in valid_set])
Y_prob_valid = model.predict(valid_set)
y_pred_valid = np.argmax(Y_prob_valid, axis=1)
utils.print_confusion_matrix(y_valid, y_pred_valid, classes2idx.keys())

baseline_classe = np.argmax(np.bincount(y))

# Test set.
X_test = tlinks_test.source_text + ' ' + tlinks_test.related_text + ' ' + tlinks_test.context
y_test = [classes2idx[cl] for cl in tlinks_test.relType]
Y_proba_test = model.predict(X_test)
y_pred_test = np.argmax(Y_proba_test, axis=1)
print(confusion_matrix(y_test, y_pred_test))

# Compute temporal-awareness.
tlinks_test['relPredicted'] = [idx2classes[idx] for idx in y_pred_test]
tlinks_test['baseline'] = idx2classes[baseline_classe]

annotations_test = tc.get_annotations(tlinks_test, ['source', 'relatedTo', 'relType'])
annotations_pred = tc.get_annotations(tlinks_test, ['source', 'relatedTo', 'relPredicted'])
annotations_baseline = tc.get_annotations(tlinks_test, ['source', 'relatedTo', 'baseline'])

ta = tc.multifile_temporal_awareness(annotations_test, annotations_pred)
ta_baseline = tc.multifile_temporal_awareness(annotations_test, annotations_baseline)

print(f"Temporal awareness baseline: {ta_baseline:.3}")
print(f"Temporal awareness model: {ta:.3}")
