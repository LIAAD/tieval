from text2timeline.datasets.custom import DatasetReader
from text2timeline import toolbox

import numpy as np
import collections

import transformers

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pprint import pprint

import random

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
        X = []
        relations = []

        for doc in dataset.docs:
            for tlink in doc.tlinks:

                # sentence(s) between source and target tokens aka context
                tokens, exp_idxs = self.get_context(doc, tlink)

                X += [(tokens, exp_idxs['src'], exp_idxs['tgt'])]
                relations += [tlink.interval_relation]

        y = self.target.indexes(relations)
        return X, y

    @staticmethod
    def get_context(doc, tlink):
        # get the context between the two expressions.
        # if the source or target are the DCT ('t0') we use as context the sentence where the other term
        # appeared. else the context is the tokens between each of the expressions.
        if tlink.source.id == doc.dct.id:

            s_context, e_context = tlink.target.endpoints
            [(s_sent, e_sent)] = [(s, e) for s, e, _ in doc.sentences if s_context >= s and e_context <= e]

            tgt = (tlink.target.endpoints[0], tlink.target.endpoints[1], tlink.target.text)

            tokens = [(s_tkn, e_tkn, tkn)
                      for s_tkn, e_tkn, tkn in doc.tokens
                      if s_tkn >= s_sent and e_tkn <= e_sent]

            idxs = collections.defaultdict(list)
            idxs['src'] = [-1]
            for idx, tkn in enumerate(tokens):
                if tgt[0] <= tkn[0] < tgt[1]:
                    idxs['tgt'] += [idx]

            return [tkn for _, _, tkn in tokens], idxs

        elif tlink.target.id == doc.dct.id:

            s_context, e_context = tlink.source.endpoints
            [(s_sent, e_sent)] = [(s, e) for s, e, _ in doc.sentences if s_context >= s and e_context <= e]

            src = (tlink.source.endpoints[0], tlink.source.endpoints[1], tlink.source.text)

            tokens = [(s_tkn, e_tkn, tkn)
                      for s_tkn, e_tkn, tkn in doc.tokens
                      if s_tkn >= s_sent and e_tkn <= e_sent]

            idxs = collections.defaultdict(list)
            idxs['tgt'] = [-1]
            for idx, tkn in enumerate(tokens):
                if src[0] <= tkn[0] < src[1]:
                    idxs['src'] += [idx]

            return [tkn for _, _, tkn in tokens], idxs

        else:

            s_context = min(tlink.source.endpoints[0], tlink.target.endpoints[0])
            e_context = max(tlink.source.endpoints[1], tlink.target.endpoints[1])

            s_sent = [s_sent for s_sent, _, _ in doc.sentences if s_sent <= s_context][-1]
            e_sent = [e_sent for _, e_sent, _ in doc.sentences if e_context <= e_sent][0]

            src = (tlink.source.endpoints[0], tlink.source.endpoints[1], tlink.source.text)
            tgt = (tlink.target.endpoints[0], tlink.target.endpoints[1], tlink.target.text)

            tokens = [(s_tkn, e_tkn, tkn)
                      for s_tkn, e_tkn, tkn in doc.tokens
                      if s_tkn >= s_sent and e_tkn <= e_sent]

            idxs = collections.defaultdict(list)
            for idx, tkn in enumerate(tokens):
                if src[0] <= tkn[0] < src[1]:
                    idxs['src'] += [idx]

                elif tgt[0] <= tkn[0] < tgt[1]:
                    idxs['tgt'] += [idx]

            return [tkn for _, _, tkn in tokens], idxs

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

        encoder = transformers.AutoModel.from_pretrained(name)
        for param in encoder.parameters():
            param.requires_grad = False
        self.encoder = encoder
        # self.gru = nn.GRU(max_length, 256, bidirectional=True)
        self.dense1 = nn.Linear(768, 768)
        self.dense2 = nn.Linear(768, 768)
        self.dense3 = nn.Linear(768, 768)
        self.classifier = nn.Linear(768, 4)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = torch.cat([i['input_ids'] for i in inputs])
        mask = torch.cat([i['attention_mask'] for i in inputs])

        x = self.encoder(x)['pooler_output']
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dense2(x)
        x = self.relu(x)
        x = self.dense3(x)
        x = self.relu(x)
        return self.classifier(x)


class LinkDataset(torch.utils.data.Dataset):

    def __init__(self, x, y, name):
        self.x = x
        self.y = y
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(name)

    def _encode(self, x):
        tokens, src, tgt = x

        # src+1 because the token [CLS] is added at the start of the sentence
        src = [s+1 for s in src]

        inputs = self.tokenizer(
            text=tokens,
            is_split_into_words=True,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=max_length
        )

        inputs['src'] = src
        inputs['tgt'] = tgt

        return inputs

    def __getitem__(self, idx):

        inputs = self._encode(self.x[idx])
        return inputs, self.y[idx]

    def __len__(self):
        return len(self.y)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# model hyperparameters
name = 'neuralmind/bert-base-portuguese-cased'
epochs = 5
lr = 1e-3
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

#train_dl = dataloader(X_train, y_train, batch_size, shuffle=True)
#valid_dl = dataloader(X_valid, y_valid, batch_size)
#test_dl = dataloader(X_test, y_test, batch_size)

train_set = LinkDataset(X_train, y_train, name)
valid_set = LinkDataset(X_valid, y_valid, name)
test_set = LinkDataset(X_test, y_test, name)


model = Model(name)
model.to(device)

loss_fn = F.cross_entropy
optimizer = optim.Adam(params=model.parameters(), lr=lr)

# train loop


def train():
    model.train()
    running_loss = 0.0
    running_acc = 0

    size = len(train_set)
    order = np.random.permutation(size)
    limits = list(range(0, size, batch_size))
    batches = [order[limits[i]: limits[i+1]] for i in range(len(limits) - 1)]
    num_batches_train = len(batches)

    for batch, idxs in enumerate(batches):
        train_batch = [train_set[idx] for idx in idxs]

        inputs_b = [x for x, _ in train_batch]
        yb = torch.tensor([y for _, y in train_batch])

        # Xb, yb = Xb.to(device), yb.to(device)

        optimizer.zero_grad()

        y_pred = model(inputs_b)

        loss = loss_fn(y_pred, yb)

        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        running_acc += (y_pred.argmax(dim=1) == y).sum().item()

        if (batch+1) % 20 == 0:
            l = running_loss / ((batch+1) * batch_size)
            a = running_acc / ((batch+1) * batch_size)
            msg = f"Batch: {batch+1:3d}/{num_batches_train}  Loss: {l:4f}  Acc: {a:2f}"
            print(msg)


def eval():

    model.eval()
    with torch.no_grad():
        size = len(valid_dl.dataset)
        y_proba = torch.cat([model(X.to(device)) for X, y in valid_dl])
        y = torch.cat([y for _, y in valid_dl]).to(device)

        loss = loss_fn(y_proba, y).sum().item()

        y_pred = y_proba.argmax(dim=1)
        acc = (y_pred == y).sum().item() / size

    msg = f"Valid loss: {loss:4f}  Valid acc: {acc:2f}"
    print(msg)


for epoch in range(epochs):
    print('#' * 30, f'Epoch {epoch+1}', '#' * 30)
    train()
    print('-' * 70)
    eval()
