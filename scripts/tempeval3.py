from text2story import utils
from text2story import read_xml as rxml
from text2story import temporal_closure as tc

import pandas as pd

from tqdm import tqdm

import fasttext

import nltk

import collections

import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import callbacks
from tensorflow.keras import metrics

import numpy as np

import kerastuner as kt

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from collections import Counter


pd.set_option('display.max_columns', 40)

tokenizer = nltk.word_tokenize

"""Read data."""
# Read AQUAINT dataset.
AQUAINT_PATH = r'../data/TempEval-3/Train/TBAQ-cleaned/AQUAINT'
timexs_aquaint = rxml.get_tags(AQUAINT_PATH, rxml.get_timexs)

events_aquaint = rxml.get_tags(AQUAINT_PATH, rxml.get_events)
makein_aquaint = rxml.get_tags(AQUAINT_PATH, rxml.get_makeinstance)
events_aquaint = events_aquaint.merge(makein_aquaint, left_on=['file', 'eid'], right_on=['file', 'eventID'])

tlinks_aquaint = rxml.get_tags(AQUAINT_PATH, rxml.get_tlinks)

base_aquaint = rxml.get_base(AQUAINT_PATH, tokenizer)


# Read TimeBank dataset.
TIMEBANK_PATH = r'../data/TempEval-3/Train/TBAQ-cleaned/TimeBank'
timexs_timebank = rxml.get_tags(TIMEBANK_PATH, rxml.get_timexs)

events_timebank = rxml.get_tags(TIMEBANK_PATH, rxml.get_events)
makein_timebank = rxml.get_tags(TIMEBANK_PATH, rxml.get_makeinstance)
events_timebank = events_timebank[['eid', 'class', 'text', 'file']].merge(makein_timebank, left_on=['file', 'eid'],
                                                                          right_on=['file', 'eventID'])

tlinks_timebank = rxml.get_tags(TIMEBANK_PATH, rxml.get_tlinks)

base_timebank = rxml.get_base(TIMEBANK_PATH, tokenizer)

# The training set will be the merge of TimeBank and AQUAINT.
timexs_train = pd.concat([timexs_timebank, timexs_aquaint], axis=0)
events_train = pd.concat([events_timebank, events_aquaint], axis=0)

tlinks_train = pd.concat([tlinks_timebank, tlinks_aquaint], axis=0)
tlinks_train['eventID'] = tlinks_train.eventInstanceID.fillna(tlinks_train.timeID).copy()
tlinks_train['relatedTo'] = tlinks_train.relatedToTime.fillna(tlinks_train.relatedToEventInstance).copy()

base_train = pd.concat([base_timebank, base_aquaint], axis=0)


# Read test data.
TEST_PATH = r'../data/TempEval-3/Test/TempEval-3-Platinum'
timexs_test = rxml.get_tags(TEST_PATH, rxml.get_timexs)

events_test = rxml.get_tags(TEST_PATH, rxml.get_events)
makein_test = rxml.get_tags(TEST_PATH, rxml.get_makeinstance)
events_test = events_test[['eid', 'class', 'text', 'file']].merge(makein_test, left_on=['file', 'eid'],
                                                                  right_on=['file', 'eventID'])

tlinks_test = rxml.get_tags(TEST_PATH, rxml.get_tlinks)
tlinks_test['eventID'] = tlinks_test.eventInstanceID.fillna(tlinks_test.timeID).copy()
tlinks_test['relatedTo'] = tlinks_test.relatedToTime.fillna(tlinks_test.relatedToEventInstance).copy()

# Remove relations that mean the same thing.
tlinks_test['relType'] = tlinks_test.relType.map(tc.relevant_relations)
tlinks_test.drop_duplicates(inplace=True)

base_test = rxml.get_base(TEST_PATH, tokenizer)


"""Preprocess data."""
# Prepare data to compute temporal closure.
files = tlinks_train.file.unique()
train_annotations = [tc.df_to_annotations(tlinks_train[tlinks_train.file == file], ['eventID', 'relatedTo', 'relType'])
                     for file in tqdm(files)]

# Compute temporal closure.
train_closure = [tc.temporal_closure(annotations) for annotations in tqdm(train_annotations)]

clean_train_closure = []
for file, file_tc in zip(train_annotations, train_closure):
    annot_base = set((e1, e2) for e1, e2, _ in file)
    file.update({(e1, e2, rel) for e1, e2, rel in file_tc if (e1, e2) not in annot_base})
    clean_train_closure.append(file)

# Build a dataframe from the closure info.
foo = [[file, *annotation] for file, annotations in zip(files, clean_train_closure) for annotation in annotations]
train_closure_df = pd.DataFrame(foo, columns=['file', 'eventID', 'relatedTo', 'relType'])

# Remove relations that mean the same thing.
train_closure_df['relType'] = train_closure_df.relType.map(tc.relevant_relations)
train_closure_df.drop_duplicates(inplace=True)

# Keep only one tlink between each event.
train_closure_df = train_closure_df.groupby(['file', 'eventID', 'relatedTo']).sample(n=1, random_state=123)

# Add the text to the dataframe.
expressions_train = dict(events_train.set_index(['file', 'eiid']).text)
expressions_train.update(dict(timexs_train.set_index(['file', 'tid']).text))
train_closure_df['eventText'] = train_closure_df.apply(lambda x: expressions_train[(x['file'], x['eventID'])], axis=1)
train_closure_df['relatedText'] = train_closure_df.apply(lambda x: expressions_train[(x['file'], x['relatedTo'])], axis=1)

expressions_test = dict(events_test.set_index(['file', 'eiid']).text)
expressions_test.update(dict(timexs_test.set_index(['file', 'tid']).text))
tlinks_test['eventText'] = tlinks_test.apply(lambda x: expressions_test[(x['file'], x['eventID'])], axis=1)
tlinks_test['relatedText'] = tlinks_test.apply(lambda x: expressions_test[(x['file'], x['relatedTo'])], axis=1)


"""Load fastText embeddings."""
ft = fasttext.load_model('fasttext/cc.pt.300.bin')
embed_dim = ft.get_dimension()

# Build word index dictionary.
tokens = base_train.token.tolist() + base_test.token.tolist()
word_freq = collections.Counter(base_train.token)
vocab = list(word_freq.keys()) + ['<oov>', '<dct>']
vocab_size = len(vocab)

word_idx = {word: idx + 1 for idx, word in enumerate(vocab)}
idx_word = {idx: word for word, idx in word_idx.items()}


"""Build input and output of the model."""
X1_tokens = utils.text2token(train_closure_df.eventText, word_idx, tokenizer) + \
            utils.text2token(tlinks_test.eventText, word_idx, tokenizer)
X2_tokens = utils.text2token(train_closure_df.relatedText, word_idx, tokenizer) + \
            utils.text2token(tlinks_test.relatedText, word_idx, tokenizer)

max_len_x1 = max(len(x) for x in X1_tokens)
max_len_x2 = max(len(x) for x in X2_tokens)

train_closure_df_shuf = train_closure_df.sample(frac=1).copy()  # Shuffle data.

X1_seq_train = utils.text2pad_sequence(train_closure_df_shuf.eventText.values, tokenizer, word_idx, max_len_x1)
X2_seq_train = utils.text2pad_sequence(train_closure_df_shuf.relatedText, tokenizer, word_idx, max_len_x2)
X1_seq_test = utils.text2pad_sequence(tlinks_test.eventText, tokenizer, word_idx, max_len_x1)
X2_seq_test = utils.text2pad_sequence(tlinks_test.relatedText, tokenizer, word_idx, max_len_x2)

# Build target.
classes = train_closure_df_shuf.relType.unique()
output_size = len(classes)

classes2idx = {cl: i for i, cl in enumerate(classes)}
idx2classes = dict((i, cl) for cl, i in classes2idx.items())

y_full_train = [classes2idx[cl] for cl in train_closure_df.relType]


# Build tensorflow dataset and split data into train and validation.
full_train_size = len(X1_seq_train)

cut = round(full_train_size * 0.8)
train_set = tf.data.Dataset.from_tensor_slices(
    ((X1_seq_train[:cut], X2_seq_train[:cut]), y_full_train[:cut])).batch(32).prefetch(1)
valid_set = tf.data.Dataset.from_tensor_slices(
    ((X1_seq_train[cut:], X2_seq_train[cut:]), y_full_train[cut:])).batch(32).prefetch(1)


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
                hp.Int('num_units', 500, 2000),
                activation='relu')(z)
            z = layers.Dropout(
                hp.Float('dropout_rate', 0, 0.5))(z)
        output = layers.Dense(output_size, activation='softmax')(z)
        model = models.Model(inputs=[input1, input2], outputs=[output])

        model.compile(
            optimizer=optimizers.Adam(lr=hp.Float('lr', 1e-3, 1e-1, sampling='log')),
            loss=tf.keras.losses.SparseCategoricalCrossentropy()
        )
        return model


model_trainable_embed = ModelBuilder(vocab, embeddings_trainable=True)
model_not_trainable_embed = ModelBuilder(vocab, embeddings_trainable=False)

hp = kt.HyperParameters()
hp.Fixed('input_operation', 'gru')
hp.Fixed('num_dense_layers', 1)
hp.Fixed('num_units', 256)
hp.Fixed('dropout_rate', 0)
hp.Fixed('lr', 1e-3)

# Hyperparameter optimization.
tuner = kt.RandomSearch(
    model_trainable_embed,
    objective='val_loss',
    max_trials=1,
    directory='models',
    project_name='test_rand_sch_trainable_embeddings_closure_3',
    hyperparameters=hp
)

early_stop_cb = callbacks.EarlyStopping(patience=2, verbose=1, restore_best_weights=True)
reduce_lr_cb = callbacks.ReduceLROnPlateau(patience=1, verbose=1, min_lr=1E-6)


# Compute the weight to assign to each class (given that the dataset is imbalanced).
y_train = np.concatenate([y for x, y in train_set])
class_count = Counter(y_train)
total = len(y_train)
num_classes = len(class_count)
class_weight = dict([(cl, total / (count * num_classes)) for cl, count in class_count.items()])

tuner.search(
    train_set,
    validation_data=valid_set,
    epochs=100,
    callbacks=[early_stop_cb, reduce_lr_cb],
    class_weight=class_weight
)


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
utils.print_confusion_matrix(y_valid, y_pred_valid, classes2idx.keys())

baseline_classe = np.argmax(np.bincount(y_full_train))

# Test set.
y_test = [classes2idx[cl] for cl in tlinks_test.relType]
Y_proba_test = best_model.predict([X1_seq_test, X2_seq_test])
y_pred_test = np.argmax(Y_proba_test, axis=1)
utils.print_confusion_matrix(y_test, y_pred_test, classes2idx.keys())

# Compute temporal-awareness.
tlinks_test['relPredicted'] = [idx2classes[idx] for idx in y_pred_test]
tlinks_test['baseline'] = idx2classes[baseline_classe]
annotations_test = [tc.df_to_annotations(tlinks_test[tlinks_test.file == file], ['eventID', 'relatedTo', 'relType'])
                    for file in tlinks_test.file.unique()]
annotations_pred = [tc.df_to_annotations(tlinks_test[tlinks_test.file == file], ['eventID', 'relatedTo', 'relPredicted'])
                    for file in tlinks_test.file.unique()]

annotations_baseline = [tc.df_to_annotations(tlinks_test[tlinks_test.file == file], ['eventID', 'relatedTo', 'baseline'])
                        for file in tlinks_test.file.unique()]

ta = tc.multifile_temporal_awareness(annotations_test, annotations_pred)
ta_baseline = tc.multifile_temporal_awareness(annotations_test, annotations_baseline)

print(f"Temporal awareness baseline: {ta_baseline:.3}")
print(f"Temporal awareness model: {ta:.3}")
