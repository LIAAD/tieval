import text2story as t2s
from text2story import read_xml as rxml

import nltk

import fasttext.util

from collections import Counter

import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import callbacks

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

import matplotlib as mpl

import numpy as np

import pandas as pd

import kerastuner as kt

mpl.use('Agg')
pd.set_option('display.max_columns', 30)

WHERE = 'local'  # server or local


"""Read data."""
if WHERE == 'local':
    train_path = r'../data/TimeBankPT/train'
    test_path = r'../data/TimeBankPT/test'
elif WHERE == 'server':
    train_path = r'/home/dock/temporal_identification/data/TimeBankPT/train'
    test_path = r'/home/dock/temporal_identification/data/TimeBankPT/test'


tokenizer = nltk.word_tokenize

# Read train data.
base_train = rxml.get_base(train_path, tokenizer)

events_train = rxml.get_tags(train_path, rxml.get_events)
timex_train = rxml.get_tags(train_path, rxml.get_timexs)
tlinks_train = rxml.get_tags(train_path, rxml.get_tlinks)

tlinks_train['relatedTo'] = tlinks_train.relatedToTime.fillna(tlinks_train.relatedToEvent).copy()
tlinks_train = t2s.utils.add_text_to_tlinks(tlinks_train, events_train, timex_train)
tlinks_train.loc[tlinks_train.task == 'B', 'relatedToken'] = '<dct>'  # Use "<dct>" as token for task B


# Read test data.
base_test = rxml.get_base(test_path, tokenizer)

events_test = rxml.get_tags(test_path, rxml.get_events)
timex_test = rxml.get_tags(test_path, rxml.get_timexs)
tlinks_test = rxml.get_tags(test_path, rxml.get_tlinks)

tlinks_test['relatedTo'] = tlinks_test.relatedToTime.fillna(tlinks_test.relatedToEvent).copy()
tlinks_test = t2s.utils.add_text_to_tlinks(tlinks_test, events_test, timex_test)
tlinks_test.loc[tlinks_test.task == 'B', 'relatedToken'] = '<dct>'  # Use "<dct>" as token for task B


"""Load fastText embeddings."""
ft = fasttext.load_model('fasttext/cc.pt.300.bin')
embed_dim = ft.get_dimension()

# Check coverage of fastText embeddings.
tokens = base_train.token.tolist() + base_test.token.tolist()
word_freq = Counter(tokens)

missing_tokens = [w for w in word_freq if w not in ft.words]
freq_miss_tokens = [word_freq[t] for t in missing_tokens]

coverage = 1 - sum(freq_miss_tokens) / len(tokens)
print(f'Coverage of fastText emebaddings: {coverage * 100:.3} %')


# Build word index dictionary.
vocab = list(word_freq.keys()) + ['<oov>', '<dct>']
vocab_size = len(vocab)

word_idx = {word: idx + 1 for idx, word in enumerate(vocab)}
idx_word = {idx: word for word, idx in word_idx.items()}


"""Build input and output of the model."""
tlinks_train_shuf = tlinks_train.sample(frac=1).copy()  # Shuffle data.

X1_tokens = t2s.utils.text2token(tlinks_train_shuf.eventToken) + t2s.utils.text2token(tlinks_test.eventToken)
X2_tokens = t2s.utils.text2token(tlinks_train_shuf.relatedToken) + t2s.utils.text2token(tlinks_test.relatedToken)

max_len_x1 = max(len(x) for x in X1_tokens)
max_len_x2 = max(len(x) for x in X2_tokens)

X1_seq_train = t2s.utils.text2pad_sequence(tlinks_train_shuf.eventToken, max_len_x1)
X2_seq_train = t2s.utils.text2pad_sequence(tlinks_train_shuf.relatedToken, max_len_x2)
X1_seq_test = t2s.utils.text2pad_sequence(tlinks_test.eventToken, max_len_x1)
X2_seq_test = t2s.utils.text2pad_sequence(tlinks_test.relatedToken, max_len_x2)

# Build target.
classes2idx = {'AFTER': 0, 'BEFORE': 1, 'BEFORE-OR-OVERLAP': 2, 'OVERLAP': 3, 'OVERLAP-OR-AFTER': 4, 'VAGUE': 5}
idx2classes = dict((i, cl) for cl, i in classes2idx.items())
y_full_train = [classes2idx[cl] for cl in tlinks_train_shuf.relType]


# Build tensorflow dataset and split data into train and validation.
full_train_size = len(X1_seq_train)

cut = round(full_train_size * 0.8)
train_set = tf.data.Dataset.from_tensor_slices(
    ((X1_seq_train[:cut], X2_seq_train[:cut]), y_full_train[:cut])).batch(128).prefetch(1)
valid_set = tf.data.Dataset.from_tensor_slices(
    ((X1_seq_train[cut:], X2_seq_train[cut:]), y_full_train[cut:])).batch(128).prefetch(1)


"""Model."""


class ModelBuilder(kt.HyperModel):
    def __init__(self, vocab, embeddings_trainable=False):
        self.embeddings_trainable = embeddings_trainable
        # Build embedding layer.
        embeddings = np.array([ft[w] for w in vocab])
        embeddings = np.vstack([np.zeros([1, embed_dim]), embeddings])  # Mask zero.
        self.embedding_init = tf.keras.initializers.Constant(embeddings)

    def build(self, hp):
        embed_layer = layers.Embedding(input_dim=vocab_size + 1, output_dim=embed_dim,
                                       embeddings_initializer=self.embedding_init,
                                       trainable=self.embeddings_trainable, name='embedding', mask_zero=True)

        input1 = layers.Input(shape=(max_len_x1,), dtype=np.int32, name='input1')
        input2 = layers.Input(shape=(max_len_x2,), dtype=np.int32, name='input2')
        z1 = embed_layer(input1)
        z2 = embed_layer(input2)
        if hp.Choice('input_operation', ['avg', 'gru']) == 'avg':
            z1 = layers.GlobalAvgPool1D()(z1)
            z2 = layers.GlobalAvgPool1D()(z2)
        else:
            z1 = layers.GRU(embed_dim, use_bias=False)(z1)
            z2 = layers.GRU(embed_dim, use_bias=False)(z2)

        z = layers.Concatenate()([z1, z2])

        for _ in range(hp.Int('num_dense_layers', 0, 3, default=1)):
            z = layers.Dense(
                hp.Int('num_units', 30, 300),
                activation='relu')(z)
            z = layers.Dropout(
                hp.Float('dropout_rate', 0, 0.5))(z)
        output = layers.Dense(6, activation='softmax')(z)
        model = models.Model(inputs=[input1, input2], outputs=[output])

        model.compile(
            optimizer=optimizers.Adam(lr=hp.Float('lr', 1e-3, 1e-2, sampling='log')),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )
        return model


model_trainable_embed = ModelBuilder(vocab, embeddings_trainable=True)
model_not_trainable_embed = ModelBuilder(vocab, embeddings_trainable=False)

# Fix hyperparameters.
hp = kt.HyperParameters()
hp.Fixed('input_operation', 'avg')
hp.Fixed('num_dense_layers', 2)
hp.Fixed('num_units', 200)
hp.Fixed('dropout_rate', 0.3)
hp.Fixed('lr', 1e-3)

# Hyperparameter optimization.
tuner = kt.RandomSearch(
    model_not_trainable_embed,
    hyperparameters=hp,
    objective='val_loss',
    max_trials=1,
    directory='models',
    project_name='rand_sch_trainable_embeddings1'
)

early_stop_cb = callbacks.EarlyStopping(patience=12, verbose=1, restore_best_weights=True)
reduce_lr_cb = callbacks.ReduceLROnPlateau(patience=5, verbose=1, min_lr=1E-6)

"""
# Compute the weight to assign to each class (given that the dataset is imbalanced).
y_train = np.concatenate([y for x, y in train_set])
class_count = Counter(y_train)
total = len(y_train)
num_classes = len(class_count)
class_weight = dict([(cl, total / (count * num_classes)) for cl, count in class_count.items()])
"""

tuner.search(
    train_set,
    validation_data=valid_set,
    epochs=100,
    callbacks=[early_stop_cb, reduce_lr_cb])


tuner.search_space_summary()
tuner.results_summary()

best_model = tuner.get_best_models(1)[0]
best_hp = tuner.get_best_hyperparameters(1)[0]
print(best_hp.values)


"""Evaluate Model."""
# Validation set.
y_valid = np.concatenate([y for x, y in valid_set])
Y_prob_valid = best_model.predict(valid_set)
y_pred_valid = np.argmax(Y_prob_valid, axis=1)
t2s.utils.print_confusion_matrix(y_valid, y_pred_valid, classes2idx.keys())
print(accuracy_score(y_valid, y_pred_valid))


# Test set.
y_test = [classes2idx[cl] for cl in tlinks_test.relType]
Y_proba_test = best_model.predict([X1_seq_test, X2_seq_test])
y_pred_test = np.argmax(Y_proba_test, axis=1)
t2s.utils.print_confusion_matrix(y_test, y_pred_test, classes2idx.keys())
print(accuracy_score(y_test, y_pred_test))

# Look at accuracy by task.
tlinks_test['relPredicted'] = [idx2classes[idx] for idx in y_pred_test]
tasks = ['A', 'B', 'C']
task_acc = [accuracy_score(tlinks_test[tlinks_test.task == task].relType,
                           tlinks_test[tlinks_test.task == task].relPredicted)
                   for task in tasks]

for task, acc in zip(tasks, task_acc):
    print(f"Accuracy of task {task}: {acc*100:.3} %")

