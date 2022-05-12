import logging
import pathlib
from typing import Iterable, List

import allennlp.modules.elmo as elmo_module
from nltk.stem import WordNetLemmatizer
import torch
import torch.nn as nn
import torch.nn.functional as F

from tieval.models.base import BaseTrainableModel
from tieval.base import Document

# elmo parameters
OPTION_FILE = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
WEIGHT_FILE = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

CSE_EMBEDDINGS_PATH = 'embeddings_0.3_200_1_timelines.txt'


class CogCompTime2(BaseTrainableModel):

    def __init__(
            self,
            model_path: str = "./models",
            resources_path: str = "./resources"
    ) -> None:

        path = pathlib.Path(path)
        self.path = model_path / "cogcomp2"
        self.resources_path = resources_path / "cogcomp2"

        self.lemmatizer = WordNetLemmatizer()

        elmo = elmo_module.Elmo(
            options_file=OPTION_FILE,
            weight_file=WEIGHT_FILE,
            num_output_representations=1,
            dropout=0
        )
        elmo.requires_grad_(requires_grad=False)

        # Common sense encoder embeddings
        self.cse_vocab = {}
        with open(CSE_EMBEDDINGS_PATH) as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                self.cse_vocab[line.split(" ")[0]] = i

        vocab_size = len(self.cse_vocab)
        cse = CommonSenseEncoder(
            vocab_size=vocab_size,
            hidden_size=120,
            emb_size=200
        )

        self.model = TemporalRelationClassifier(
            elmo=elmo,
            common_sense_encoder=cse,
            granularity=0.1,
            common_sense_emb_dim=32,
            embedding_dim=1024,
            lstm_hidden_dim=64,
            nn_hidden_dim=64,
            output_dim=4
        )

        if not self.path.is_dir():
            self.download()

        self.load()

    def download(self):

        url = metadata.MODELS_URL["cogcomp2"]
        utils._download_url(url, self.path.parent)

        url = metadata.MODELS_URL["cogcomp2"]
        utils._download_url(url, self.path.parent)

    def predict(self, documents: Iterable[Document]):

        data = self.data_pipeline(documents)

        predictions = []
        for doc, doc_data in data.items():

            logging.info(f"Processing document {doc}")
            for idx, sample in enumerate(doc_data):

                if idx % 20 == 0:
                    logging.info(f"Sample {idx}/{len(doc_data)}")

                pred = self.model(
                    entities_idxs=sample["entities_idxs"],
                    lemma_ids=sample["lemma_ids"],
                    context_tokens=sample["context_tokens"],
                )

                predictions += [pred]

        return predictions

    def fit(self, documents: Iterable[Document]):
        pass

    def save(self) -> None:
        checkpoint = self.model.state_dict()
        torch.save(checkpoint, self.path)

    def load(self) -> None:
        checkpoint = torch.load(self.path)
        self.model.load_state_dict(checkpoint)

    def data_pipeline(self, documents: Iterable[Document]):

        data = {}
        for doc in documents:
            doc_data = []
            for tlink in doc.tlinks:
                src = tlink.source
                tgt = tlink.target

                context = []
                tlink_endpoints = [min(src.endpoints + tgt.endpoints),
                                   max(src.endpoints + tgt.endpoints)]
                s_tl, e_tl = tlink_endpoints
                for sent in doc.sentences:
                    s_sent, e_sent = sent.span

                    cond_1 = s_sent <= s_tl <= e_sent
                    cond_2 = s_sent <= e_tl <= e_sent
                    cond_3 = s_tl <= s_sent < e_sent <= e_tl
                    if cond_1 or cond_2 or cond_3:
                        context += sent.tokens

                # source and target indexes
                src_idx, tgt_idx = None, None
                for idx, tkn in enumerate(context):
                    if tkn.span == list(src.endpoints):
                        src_idx = idx
                    elif tkn.span == list(tgt.endpoints):
                        tgt_idx = idx

                # retrieve elmo character ids of context sentence(s)
                context_tokens = [tkn.content for tkn in context]

                # transform lemma of source and target events into vocab ids
                src_lemma = self.lemmatizer.lemmatize(src.text)
                tgt_lemma = self.lemmatizer.lemmatize(tgt.text)
                src_lemma_id = self.cse_vocab.get(src_lemma, 0)
                tgt_lemma_id = self.cse_vocab.get(tgt_lemma, 0)

                if src_idx is None or tgt_idx is None:
                    logging.info(f"Tlink {tlink} is missing something.")
                    continue

                doc_data += [{
                    "entities_idxs": [src_idx, tgt_idx],
                    "lemma_ids": [src_lemma_id, tgt_lemma_id],
                    "context_tokens": context_tokens,
                }]

            data[doc.name] = doc_data

        return data


class TemporalRelationClassifier(nn.Module):

    def __init__(
            self,
            elmo,
            common_sense_encoder,
            granularity,
            common_sense_emb_dim,
            embedding_dim,
            lstm_hidden_dim,
            nn_hidden_dim,
            output_dim
    ):
        super(TemporalRelationClassifier, self).__init__()

        self.granularity = granularity

        self.cse = common_sense_encoder
        self.elmo = elmo

        self.common_sense_emb = nn.Embedding(
            int(1.0 / granularity),
            common_sense_emb_dim
        )

        self.lstm = nn.LSTM(
            embedding_dim,
            lstm_hidden_dim,
            num_layers=1,
            bidirectional=False
        )

        self.h_lstm2h_nn = nn.Linear(192, nn_hidden_dim)
        self.h_nn2o = nn.Linear(128, output_dim)

    def forward(
            self,
            entities_idxs: List[int],
            lemma_ids: List[int],
            context_tokens: List[str],
    ):

        # common sense module
        src_tgt_tensor = torch.LongTensor([lemma_ids])
        tgt_src_tensor = torch.LongTensor([lemma_ids[::-1]])
        encoding_src_tgt = self.cse(src_tgt_tensor)
        encoding_tgt_src = self.cse(tgt_src_tensor)
        encoding = torch.cat([encoding_src_tgt, encoding_tgt_src], 1)
        encoding = encoding.view(1, -1)

        granularity_inv = int(1.0 / self.granularity)
        idx1 = min(granularity_inv - 1, int(encoding[0][0] / self.granularity))
        idx2 = min(granularity_inv - 1, int(encoding[0][1] / self.granularity))
        idxs = torch.tensor([idx1, idx2])
        cse = self.common_sense_emb(idxs)
        cse_flat = cse.view(1, -1)

        # classifier
        elmo_character_ids = elmo_module.batch_to_ids([context_tokens])
        embeds = self.elmo(elmo_character_ids)['elmo_representations'][0][0]
        embeds = embeds.view(len(context_tokens), 1, -1)

        lstm_out, _ = self.lstm(embeds)
        lstm_out = lstm_out[[entities_idxs]]
        lstm_out_flat = lstm_out.view(1, -1)

        events_cse = torch.cat((lstm_out_flat, cse_flat), 1)
        h_nn = self.h_lstm2h_nn(events_cse)
        h_nn = F.relu(h_nn)

        h_cse = torch.cat((h_nn, cse_flat), 1)
        logits = self.h_nn2o(h_cse)

        return logits


class CommonSenseEncoder(nn.Module):
    """Network to train siamese embeddings."""

    def __init__(
            self,
            vocab_size: int,
            emb_size: int,
            hidden_size: int,
    ) -> None:
        super(CommonSenseEncoder, self).__init__()

        self.emb_layer = nn.Embedding(vocab_size, emb_size)
        self.fc1 = nn.Linear(emb_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

        self.is_training = True

    def forward(self, x):
        h = self.emb_layer(x)
        h = torch.cat((h[:, 0, :], h[:, 1, :]), dim=1)
        h = F.dropout(h, p=0.3, training=self.is_training)
        h = self.fc1(h)
        h = F.relu(h)
        h = self.fc2(h)
        y = torch.sigmoid(h)
        return y
