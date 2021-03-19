from text2story import utils
from text2story import read_xml as rxml
from text2story import temporal_closure as tc

import pandas as pd

from tqdm import tqdm

import nltk

import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import callbacks
from tensorflow.keras import metrics
from tensorflow.keras import losses

import tensorflow_hub as hub
import tensorflow_text  # Requirement for the preprocessing layer of BERT.
from official.nlp import optimization

import numpy as np

from sklearn.metrics import confusion_matrix

tf.get_logger().setLevel('ERROR')
pd.set_option('display.max_columns', 40)

tokenizer = nltk.word_tokenize

"""Read data."""
# The training set will be the merge of TimeBank and AQUAINT.
AQUAINT_PATH = r'../data/TempEval-3/Train/TBAQ-cleaned/AQUAINT'
TIMEBANK_PATH = r'../data/TempEval-3/Train/TBAQ-cleaned/TimeBank'

base_aquaint, tlinks_aquaint = rxml.read_tempeval3(AQUAINT_PATH, tokenizer)
base_timebank, tlinks_timebank = rxml.read_tempeval3(TIMEBANK_PATH, tokenizer)

base = pd.concat([base_timebank, base_aquaint], axis=0)
tlinks = pd.concat([tlinks_timebank, tlinks_aquaint], axis=0)

# Read test data.
TEST_PATH = r'../data/TempEval-3/Test/TempEval-3-Platinum'
base_test, tlinks_test = rxml.read_tempeval3(TEST_PATH, tokenizer)

# Remove relations that mean the same thing.
tlinks_test['relType'] = tlinks_test.relType.map(tc.relevant_relations)
tlinks_test.drop_duplicates(inplace=True)

"""Preprocess data."""
# Prepare data to compute temporal closure.
annotations = tc.get_annotations(tlinks)
closure = tc.get_temporal_closure(annotations)

# Build a dataframe from the closure info.
files = tlinks.file.unique()
foo = [[file, *annotation] for file, annotations in zip(files, closure) for annotation in annotations]
tlinks_closure = pd.DataFrame(foo, columns=['file', 'source', 'relatedTo', 'relType'])

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

tlinks_closure['relType'] = tlinks_closure.relType.map(reduce_relations)
tlinks_closure.drop_duplicates(inplace=True)

# Keep only one tlink between each pair.
tlinks_closure = tlinks_closure.groupby(['file', 'source', 'relatedTo']).sample(n=1, random_state=123)

# Add text for each token and the context between them to tlinks_closure dataframe.
tlinks_closure = tlinks_closure.merge(base, left_on=['file', 'source'], right_on=['file', 'tag_id'], how='left')
tlinks_closure = tlinks_closure.merge(base, left_on=['file', 'relatedTo'], right_on=['file', 'tag_id'])
tlinks_closure.rename(columns={
    'sentence_x': 'source_sent',
    'token_x': 'source_text',
    'sentence_y': 'related_sent',
    'token_y': 'related_text'}, inplace=True)
tlinks_closure.drop(['tag_id_x', 'tag_id_y'], axis=1, inplace=True)

# Limit the temporal links to be inside the same sentece, or betwen neighbor sentences.
sent_distance = tlinks_closure.related_sent - tlinks_closure.source_sent
tlinks_closure = tlinks_closure[(tlinks_closure.source == 't0') |
                                (tlinks_closure.relatedTo == 't0') |
                                (abs(sent_distance) <= 1)]

# Add context.
tlinks_closure = utils.add_context(tlinks_closure, base)

# Build inputs for the model.
X_source = tlinks_closure.source_text.values
X_related = tlinks_closure.related_text.values
X_context = tlinks_closure.context.values
X = tlinks_closure.source_text + ' ' + tlinks_closure.related_text + ' ' + tlinks_closure.context

# input_len = [len(tokenizer(x)) for x in tqdm(X)]

# Build target.
classes = tlinks_closure.relType.unique()
output_size = len(classes)

classes2idx = {cl: i for i, cl in enumerate(classes)}
idx2classes = dict((i, cl) for cl, i in classes2idx.items())

y = [classes2idx[cl] for cl in tlinks_closure.relType]

# Split data into train and validation and build a tensorflow dataset.
data_size = len(tlinks_closure)
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
    z = encoder(encoder_input)['pooled_output']
    z = layers.Dense(126, activation='relu')(z)
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

model_path = r'bert_L-4_H-256_A-4_based_model_context_reduce_labels'
checkpoint_cb = callbacks.ModelCheckpoint(model_path, save_best_only=True)
early_stop_cb = callbacks.EarlyStopping(patience=2, verbose=1, restore_best_weights=True)
reduce_lr_cb = callbacks.ReduceLROnPlateau(patience=1, verbose=1, min_lr=1E-6)

model.fit(train_set, validation_data=valid_set, epochs=epochs, callbacks=[early_stop_cb, checkpoint_cb])

model = models.load_model(model_path)


"""Evaluate Model."""
# Validation set.
y_valid = np.concatenate([y for x, y in valid_set])
Y_prob_valid = model.predict(valid_set)
y_pred_valid = np.argmax(Y_prob_valid, axis=1)
utils.print_confusion_matrix(y_valid, y_pred_valid, classes2idx.keys())

baseline_classe = np.argmax(np.bincount(y))

# Test set.
y_test = [classes2idx[cl] for cl in tlinks_test.relType]
Y_proba_test = model.predict([tlinks_test.eventText, tlinks_test.relatedText])
y_pred_test = np.argmax(Y_proba_test, axis=1)
confusion_matrix(y_test, y_pred_test)

# Compute temporal-awareness.
tlinks['relPredicted'] = [idx2classes[idx] for idx in y_pred_test]
tlinks['baseline'] = idx2classes[baseline_classe]
annotations_test = [tc.df_to_annotations(tlinks[tlinks.file == file], ['eventID', 'relatedTo', 'relType'])
                    for file in tlinks.file.unique()]
annotations_pred = [tc.df_to_annotations(tlinks[tlinks.file == file], ['eventID', 'relatedTo', 'relPredicted'])
                    for file in tlinks.file.unique()]

annotations_baseline = [tc.df_to_annotations(tlinks[tlinks.file == file], ['eventID', 'relatedTo', 'baseline'])
                        for file in tlinks.file.unique()]

ta = tc.multifile_temporal_awareness(annotations_test, annotations_pred)
ta_baseline = tc.multifile_temporal_awareness(annotations_test, annotations_baseline)

print(f"Temporal awareness baseline: {ta_baseline:.3}")
print(f"Temporal awareness model: {ta:.3}")

