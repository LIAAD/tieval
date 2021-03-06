import nltk

import os

import numpy as np

import itertools

from tqdm import tqdm

import collections

import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import callbacks
from tensorflow.keras import metrics

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import pandas as pd

pd.set_option("display.max_columns", 30)

"""Read data."""
train_path = r"data/train/english/data"

base = pd.read_csv(f"{train_path}/base-segmentation.tab", sep="\t",
                   names=["doc", "line", "position", "token"])

# Because the line column has some missing lines (for example in doc wsj_0610 doesn't have line 18 but has line 19...)
# that problem is fixed in the following lines.
base.sort_values(["doc", "line", "position"], inplace=True)
base["line_diff"] = base.line.diff().apply(lambda x: 0 if x <= 1 else x - 1).fillna(0)
base["line_diff"] = base.groupby("doc").line_diff.cumsum()
base["old_line"] = base.line.copy()
base["line"] = base.line - base.line_diff

base.drop(["line_diff"], inplace=True, axis=1)

docs = base.doc.unique()
docs_id = [f"doc{i:03}" for i in range(len(docs))]
map_docs_id = pd.Series(dict(zip(docs, docs_id)))

dct = pd.read_csv(f"{train_path}/dct.txt", sep="\t", names=["doc", "date"])
dct["date"] = pd.to_datetime(dct.date, format="%Y%m%d")

# Load events and temporal expressions.
colnames_exten = ["doc", "old_line", "position", "tag", "tag_id", "instance_id"]
event_exten = pd.read_csv(f"{train_path}/event-extents.tab", sep="\t", names=colnames_exten)
timex_exten = pd.read_csv(f"{train_path}/timex-extents.tab", sep="\t", names=colnames_exten)

event_timex = pd.concat([event_exten, timex_exten])
event_timex.sort_values(["doc", "old_line", "position"], inplace=True)

# Fix the missing lines problem.
event_timex = event_timex.merge(
    base[["doc", "line", "old_line"]].drop_duplicates(),
    how="left"
)

# Creat unique tag for each event/ timex.
event_timex["doc_id"] = event_timex.doc.map(map_docs_id)
event_timex["tag_uid"] = event_timex["doc_id"] + "_" + event_timex["tag_id"]

event_timex.drop(["tag_id", "instance_id", "doc_id", "old_line"], axis=1, inplace=True)

# Load temporal links.
colnames_tlinks = ["doc", "tag_id_1", "tag_id_2", "tlink"]
tlinks_dct = pd.read_csv(f"{train_path}/tlinks-dct-event.tab", sep="\t", names=colnames_tlinks)
tlinks_main = pd.read_csv(f"{train_path}/tlinks-main-events.tab", sep="\t", names=colnames_tlinks)
tlinks_sub = pd.read_csv(f"{train_path}/tlinks-subordinated-events.tab", sep="\t", names=colnames_tlinks)
tlinks_timex = pd.read_csv(f"{train_path}/tlinks-timex-event.tab", sep="\t", names=colnames_tlinks)

tlinks = pd.concat([tlinks_main, tlinks_sub, tlinks_timex])

# Creat unique tag for tlinks.
tlinks["doc_id"] = tlinks.doc.map(map_docs_id)
tlinks["tag_uid_1"] = tlinks["doc_id"] + "_" + tlinks["tag_id_1"]
tlinks["tag_uid_2"] = tlinks["doc_id"] + "_" + tlinks["tag_id_2"]

tlinks.drop(["tag_id_1", "tag_id_2", "doc_id"], axis=1, inplace=True)

# Find all the possible combinations of event and temporal expression.
event_timex_by_doc = event_timex.groupby("doc").tag_uid.unique()

event_timex_comb = [[(e1, e2) for e1, e2 in list(itertools.combinations(doc, 2))
                     if not ((e1[0] == "t") and (e2[0] == "t"))]
                    for doc in event_timex_by_doc]

# Build a dataframe of all the combinations and the temporal link if any.
df = [(doc, tag1, tag2) for doc, comb in zip(docs, event_timex_comb) for tag1, tag2 in comb]
df = pd.DataFrame(df, columns=["doc", "tag_uid_1", "tag_uid_2"])
df = df.merge(tlinks, how="outer")

# Add the text between the two tags to the dataframe.
base = base.merge(event_timex[["doc", "line", "position", "tag_uid"]], how="left")

# Concatenate line positions into lists.
positions = event_timex.groupby("tag_uid").position.unique()
event_timex.drop("position", axis=1, inplace=True)
event_timex.drop_duplicates(inplace=True)
event_timex["position"] = event_timex.tag_uid.map(positions)

df = df.merge(
    event_timex[["tag_uid", "line", "position"]],
    left_on=["tag_uid_1"],
    right_on=["tag_uid"],
    how="left"
)

df.drop(["tag_uid"], inplace=True, axis=1)

df = df.merge(
    event_timex[["tag_uid", "line", "position"]],
    left_on=["tag_uid_2"],
    right_on=["tag_uid"],
    how="left",
    suffixes=["_1", "_2"]
)

df.drop(["tag_uid"], inplace=True, axis=1)


"""Data analysis."""
# Line difference between annoted links.
df["line_dif"] = df["line_2"] - df["line_1"]
# df[~df.tlink.isna()].line_dif.value_counts()
df = df[abs(df["line_dif"]) < 3]

# Add the sentences between the two events.
corpus_by_line = base.groupby(["doc", "line"]).token.apply(lambda x: x.tolist())
corpus_by_line = corpus_by_line.to_dict()

sentences = []
for idx, row in df.iterrows():
    l1, l2 = int(row.line_1), int(row.line_2)
    lines = list(range(l1, l2 + 1)) if l1 <= l2 else list(range(l2, l1 + 1))
    sent = []
    for line in lines:
        sent += corpus_by_line[(row.doc, line)]
    sentences.append(sent)

df["sentences"] = sentences

df["target"] = ~df.tlink.isna()

"""Prepare model for modeling."""
# Load GloVe vectors.
glove_embeddings = dict()
with open(r"data\glove.6B\glove.6B.50d.txt", encoding="utf_8") as f:
    for line in f.readlines():
        line = line.strip().split()
        glove_embeddings[line[0]] = np.array(line[1:], dtype=np.float)

vocab_size = len(glove_embeddings)
embed_dim = len(glove_embeddings["hi"])

# Compute coverage of GloVe vectors.
vocab = set(token for sent in sentences for token in sent)

oov_tokens = [token for token in vocab if token.lower() not in glove_embeddings]
token_count = collections.Counter(base.token)

oov_tokens_num = sum([token_count[token] for token in oov_tokens])

oov_perc = len(oov_tokens) / len(token_count)
coverage = 1 - oov_tokens_num / len(base.token)

print(f"GloVe has a coverage of {coverage * 100:.4}%.")

# From tokens to sequence.
token_idx = dict([(token, idx + 1) for idx, token in enumerate(glove_embeddings)])
idx_token = dict([(idx + 1, token) for idx, token in enumerate(glove_embeddings)])

sequences = [[token_idx[t.lower()] for t in sent if t.lower() in glove_embeddings] for sent in sentences]

maxlen = max(len(seq) for seq in sequences)

# Pad sequences.
X = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen, padding="post")
y = df.target.values

with open("X.npy", "wb") as f:
    np.save(f, X)

with open("y.npy", "wb") as f:
    np.save(f, y)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, stratify=y, random_state=123, test_size=0.2, shuffle=True)

train_set = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(64).prefetch(1)
valid_set = tf.data.Dataset.from_tensor_slices((X_valid, y_valid)).batch(64).prefetch(1)

"""Model."""
glove_embed_matrix = np.array(list(glove_embeddings.values()))
embed_matrix = tf.keras.initializers.Constant(glove_embed_matrix)

model = models.Sequential([
    layers.Embedding(vocab_size, embed_dim, input_length=maxlen, embeddings_initializer=embed_matrix,
                     trainable=False, mask_zero=True),
    layers.GRU(256, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
    layers.GRU(128, return_sequences=False),
    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=optimizers.Adam(learning_rate=0.1),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

reduce_lr_cb = callbacks.ReduceLROnPlateau(patience=2, min_lr=10e-6, verbose=2)
early_stop_cb = callbacks.EarlyStopping(patience=4, restore_best_weights=True)

class_weight = {1: 1 - sum(y_train) / len(y_train), 0: sum(y_train) / len(y_train)}

history = model.fit(
    train_set,
    epochs=50,
    validation_data=valid_set,
    callbacks=[reduce_lr_cb, early_stop_cb],
    class_weight=class_weight
)

y_pred = model.predict(valid_set)

accuracy_score(y_valid, y_pred > 0.5)
confusion_matrix(y_valid, y_pred > 0.5)
precision_score(y_valid, y_pred > 0.5)

model.summary()

history