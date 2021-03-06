import tensorflow as tf

from sklearn.metrics import confusion_matrix

import pandas as pd

from typing import Dict
from typing import List

from tqdm import tqdm


def text2token(text_list, word_idx: Dict, tokenizer) -> List:
    token_list = [[token if token in word_idx else '<oov>' for token in tokenizer(text)]
                  if text != '<dct>' else ['<dct>']
                  for text in text_list]
    return token_list


def tokens2sequence(token_list: List, word_idx: Dict) -> List:
    return [[word_idx[token] for token in tokens] for tokens in token_list]


def text2pad_sequence(list_text: List, tokenizer, word_idx, pad_len: int = None) -> List:
    tokens = text2token(list_text, word_idx, tokenizer)
    sequences = tokens2sequence(tokens, word_idx)

    if pad_len:
        pad_seq = tf.keras.preprocessing.sequence.pad_sequences(sequences, pad_len, padding='post')
    else:
        max_len = max(len(seq) for seq in sequences)
        pad_seq = tf.keras.preprocessing.sequence.pad_sequences(sequences, max_len, padding='post')
    return pad_seq


def add_text_to_tlinks(tlinks_df: pd.DataFrame, events_df: pd.DataFrame, timex_df: pd.DataFrame) -> pd.DataFrame:
    tlinks_df = tlinks_df.merge(
        events_df[['eid', 'file', 'text']],
        left_on=['eventID', 'file'],
        right_on=['eid', 'file']
    )

    tlinks_df = tlinks_df.merge(
        events_df[['eid', 'file', 'text']],
        left_on=['relatedTo', 'file'],
        right_on=['eid', 'file'],
        how='left'
    )

    tlinks_df = tlinks_df.merge(
        timex_df[['tid', 'file', 'text']],
        left_on=['relatedTo', 'file'],
        right_on=['tid', 'file'],
        how='left'
    )

    tlinks_df['relatedToken'] = tlinks_df.text_y.fillna(tlinks_df.text).copy()

    tlinks_df.drop(['eid_x', 'eid_y', 'tid', 'text_y', 'text'], inplace=True, axis=1)
    tlinks_df.rename(columns={'text_x': 'eventToken'}, inplace=True)

    return tlinks_df


def add_context(tlinks, base):
    file_sentence = base.groupby(['file', 'sentence']).token.apply(' '.join)
    context_sent = []
    for _, row in tqdm(tlinks.iterrows()):
        file, source_idx, related_idx = row[['file', 'source_sent', 'related_sent']]
        if source_idx == related_idx:
            context_sent.append(file_sentence[file, source_idx])
        elif source_idx > related_idx:
            context_sent.append(file_sentence[file, source_idx] + file_sentence[file, related_idx])
        elif source_idx < related_idx:
            context_sent.append(file_sentence[file, related_idx] + file_sentence[file, source_idx])
        elif source_idx == -1:
            context_sent.append(file_sentence[file, related_idx])
        elif related_idx == -1:
            context_sent.append(file_sentence[file, source_idx])
    tlinks['context'] = context_sent
    return tlinks


def add_tokens(tlinks, base):
    tlinks = tlinks.merge(base, left_on=['file', 'source'], right_on=['file', 'tag_id'], how='left')
    tlinks = tlinks.merge(base, left_on=['file', 'relatedTo'], right_on=['file', 'tag_id'])
    tlinks.rename(columns={
        'sentence_x': 'source_sent',
        'token_x': 'source_text',
        'sentence_y': 'related_sent',
        'token_y': 'related_text'}, inplace=True)
    return tlinks.drop(['tag_id_x', 'tag_id_y'], axis=1)


def print_confusion_matrix(y_true, y_pred, columns):
    cm = confusion_matrix(y_true, y_pred)
    cm = pd.DataFrame(cm, index=columns, columns=columns)
    print(cm)
