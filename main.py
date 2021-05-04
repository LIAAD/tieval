from text2timeline.datasets.custom import DatasetReader
from text2timeline import toolbox

import numpy as np
import collections
from pprint import pprint

import transformers

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt


class BuildIO:
    def __init__(self, classes):
        """Build input output from the documents.

        :param classes: The classes in the target variable. In the point classification problem the classes would be
        ['<', '>', '=']. In the interval relation would be ['BEFORE', 'AFTER', (...)].
        """
        self.target = toolbox.Target(classes)
        self.inputs = dict()
        self.outputs = dict()

    def run(self, dataset):
        """

        :param docs: The
        :return:
        """
        X_context = []
        relations = []

        for doc in dataset.docs:
            for tlink in doc.tlinks:

                # sentence(s) between source and target tokens aka context
                context = self.get_context(doc, tlink)

                X_context.append(context)
                relations.append(tlink.interval_relation)

        y = self.target.indexes(relations)
        # y = self.target.one_hot(y_idxs)

        return X_context, y

    @staticmethod
    def get_context(doc, tlink):
        # get the context between the two expressions.
        # if the source or target are the DCT ('t0') we use as context the sentence where the other term
        # appeared. else the context is the tokens between each of the expressions.
        if tlink.source.id == doc.dct.id:

            s_context, e_context = tlink.target.endpoints
            [(s_sent, e_sent)] = [(s, e) for s, e, _ in doc.sentences if s_context >= s and e_context <= e]

            tgt = (tlink.target.endpoints[0], tlink.target.endpoints[1], tlink.target.text)

            tokens = []
            for s_tkn, e_tkn, tkn in doc.tokens:
                if s_tkn >= s_sent and e_tkn <= e_sent:

                    if (s_tkn, e_tkn, tkn) == tgt:
                        tokens += ['<target>', tkn, '</target>']

                    else:
                        tokens += [tkn]

            return ' '.join(tokens)

        elif tlink.target.id == doc.dct.id:

            s_context, e_context = tlink.source.endpoints
            [(s_sent, e_sent)] = [(s, e) for s, e, _ in doc.sentences if s_context >= s and e_context <= e]

            src = (tlink.source.endpoints[0], tlink.source.endpoints[1], tlink.source.text)

            tokens = []
            for s_tkn, e_tkn, tkn in doc.tokens:
                if s_tkn >= s_sent and e_tkn <= e_sent:

                    if (s_tkn, e_tkn, tkn) == src:
                        tokens += ['<source>', tkn, '</source>']

                    else:
                        tokens += [tkn]

            return ' '.join(tokens)

        else:

            s_context = min(tlink.source.endpoints[0], tlink.target.endpoints[0])
            e_context = max(tlink.source.endpoints[1], tlink.target.endpoints[1])

            s_sent = [s_sent for s_sent, _, _ in doc.sentences if s_sent <= s_context][-1]
            e_sent = [e_sent for _, e_sent, _ in doc.sentences if e_context <= e_sent][0]

            src = (tlink.source.endpoints[0], tlink.source.endpoints[1], tlink.source.text)
            tgt = (tlink.target.endpoints[0], tlink.target.endpoints[1], tlink.target.text)

            tokens = []
            for s_tkn, e_tkn, tkn in doc.tokens:
                if s_tkn >= s_sent and e_tkn <= e_sent:

                    if (s_tkn, e_tkn, tkn) == src:
                        tokens += ['<source>', tkn, '</source>']

                    elif (s_tkn, e_tkn, tkn) == tgt:
                        tokens += ['<target>', tkn, '</target>']

                    else:
                        tokens += [tkn]

            return ' '.join(tokens)


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

        print()


class Model(nn.Module):

    def __init__(self, name: str):
        super().__init__()
        self.name = name

        self.encoder = transformers.AutoModel.from_pretrained(name)
        self.gru = nn.GRU(max_length, 256, bidirectional=True)
        self.classifier = nn.Linear(768, 4)

    def forward(self, x):
        x = self.encoder(x)['pooler_output']
        return self.classifier(x)


class LinkDataset(torch.utils.data.Dataset):

    def __init__(self, x, y, name):
        self.x = x
        self.y = y
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(name)

    def _encode(self, txt):
        seq = self.tokenizer.encode(
            txt,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=max_length
        )
        return seq

    def __getitem__(self, idx):

        seq = self._encode(self.x[idx])
        return seq[0], self.y[idx]

    def __len__(self):
        return len(self.y)


def dataloader(x, y, batch_size, shuffle=False):
    dataset = LinkDataset(x, y, name)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# model hyperparameters
name = 'neuralmind/bert-base-portuguese-cased'
lr = 1e-5
batch_size = 2
max_length = 512


# load dataset
reader = DatasetReader()
reader.read(['timebank-pt'])

train_valid_set = reader.datasets[1]
train_set, valid_set = train_valid_set.split(0.8)
test_set = reader.datasets[0]

# reduce dataset tlinks set
map = {
    'OVERLAP': 'OVERLAP',
    'BEFORE': 'BEFORE',
    'AFTER': 'AFTER',
    'VAGUE': 'VAGUE',
    'BEFORE-OR-OVERLAP': 'VAGUE',
    'OVERLAP-OR-AFTER': 'VAGUE'
}
train_set.reduce_tlinks(map)
valid_set.reduce_tlinks(map)
test_set.reduce_tlinks(map)


train_tlinks_count = train_set.tlinks_count()
valid_tlinks_count = valid_set.tlinks_count()
test_tlinks_count = test_set.tlinks_count()

""" build model input and output """

builder = BuildIO(train_tlinks_count.keys())

X_train, y_train = builder.run(train_set)
X_valid, y_valid = builder.run(valid_set)
X_test, y_test = builder.run(test_set)


""" build model """

train_dl = dataloader(X_train, y_train, batch_size, shuffle=True)
valid_dl = dataloader(X_valid, y_valid, batch_size)
test_dl = dataloader(X_test, y_test, batch_size)


model = Model(name)

loss_fn = F.cross_entropy
optimizer = optim.Adam(params=model.parameters(), lr=lr)

# train loop
num_batches_train = len(train_dl)

model.train()
running_loss = 0.0
for batch, (X, y) in enumerate(train_dl):

    X, y = X.to(DEVICE), y.to(DEVICE)

    optimizer.zero_grad()

    y_pred = model(X)

    loss = loss_fn(y_pred, y)

    loss.backward()
    optimizer.step()

    # print statistics
    running_loss += loss
    #if num_batches_train % 10 == 0:
    msg = f"Batch: {batch:2d}/{num_batches_train}  Loss: {running_loss / (batch * batch_size)}"
    print(msg)
