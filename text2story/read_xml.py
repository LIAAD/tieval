from xml.etree import ElementTree

import pandas as pd

import glob

import os

import bs4

import nltk.data

sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


def get_text(xml_root):
    return ''.join(txt for txt in xml_root.itertext()).strip()


def get_dct(xml_root):
    dct_timex = xml_root.find(".//TIMEX3[@functionInDocument='CREATION_TIME']")

    dct = dct_timex.attrib['value']
    tag = dct_timex.attrib['tid']
    return tag, dct


def get_events(xml_root):
    # Add text to event attributes.
    for event in xml_root.findall('.//EVENT'):
        event.attrib['text'] = event.text
    return pd.DataFrame(event.attrib for event in xml_root.findall('.//EVENT'))


def get_timexs(xml_root):
    # Add text to timex attributes.
    for timex in xml_root.findall('.//TIMEX3'):
        timex.attrib['text'] = timex.text
    return pd.DataFrame(timex.attrib for timex in xml_root.findall('.//TIMEX3'))


def get_tlinks(xml_root):
    return pd.DataFrame(tlink.attrib for tlink in xml_root.findall('.//TLINK'))


def get_makeinstance(xml_root):
    return pd.DataFrame(tlink.attrib for tlink in xml_root.findall('.//MAKEINSTANCE'))


def get_tags(folder_path, get_func=get_tlinks):
    glob_path = os.path.join(folder_path, '*.tml')
    files_path = glob.glob(glob_path)

    df = []
    for path in files_path:
        root = ElementTree.parse(path).getroot()
        tags_df = get_func(root)
        tags_df['file'] = os.path.basename(path).replace('.tml', '')
        df.append(tags_df)

    df = pd.concat(df)
    return df.reset_index(drop=True)


def read_xml_with_bs4(path):
    with open(path, encoding='utf-8') as f:
        content = f.readlines()
        content = ''.join(content)
        bs_content = bs4.BeautifulSoup(content, 'xml')
    return bs_content


def child_tag(child):
    if child.name is None:
        return None
    elif child.name.lower() == 'event':
        return child['eid']
    elif child.name.lower() == 'timex3':
        return child['tid']


def build_base(path, tokenizer):
    bs_content = read_xml_with_bs4(path)

    # Build base dataframe.
    base = {
        'token': [],
        'tag_id': [],
        'sentence': []
    }

    # Add tokens, tags and sentence index.
    sentences = bs_content.find_all('s')
    if len(sentences) == 0:
        text_tag = bs_content.find('TEXT')
        text = text_tag.renderContents().decode('utf-8').split('\n\n')
        format_sentences = ' '.join(f'<s>{sent}</s>' for sent in text if sent)
        soup = bs4.BeautifulSoup(format_sentences, 'lxml')
        sentences = soup.find_all('s')

    for idx_sent, sent in enumerate(sentences):
        childs = sent.contents
        for child in childs:
            tokens = tokenizer(child.string)
            tag = child_tag(child)

            if tag:
                base['token'] += [child.string]
                base['tag_id'] += [tag]
                base['sentence'] += [idx_sent]
            else:
                base['token'] += tokens
                base['tag_id'] += [tag] * len(tokens)
                base['sentence'] += [idx_sent] * len(tokens)

    # Add document creation time (dct) to base.
    xml_root = ElementTree.parse(path).getroot()
    dct_tag, dct = get_dct(xml_root)

    base['tag_id'] += [dct_tag]
    base['token'] += [dct]
    base['sentence'] += [-1]

    base = pd.DataFrame(base)

    return base.reset_index()


def get_base(folder_path, tokenizer):
    glob_path = os.path.join(folder_path, '*.tml')
    files_path = glob.glob(glob_path)

    base = []
    for path in files_path:
        df = build_base(path, tokenizer)
        df['file'] = os.path.basename(path).replace('.tml', '')
        base.append(df)
        if max(df[['file', 'tag_id']].value_counts())>1:
            break

    return pd.concat(base)


def read_dir(path: str, tokenizer) -> [pd.DataFrame, pd.DataFrame]:
    """
    Reads all .tml files that are in path.

    :param path: path to the folder to read.
    :param tokenizer:
    :return: two pandas DataFrames. One with all the tokens and another with the temporal links.
    """
    # Build base.
    base = get_base(path, tokenizer)
    makein = get_tags(path, get_makeinstance)
    base = base.merge(
        makein[['file', 'eventID', 'eiid']], left_on=['file', 'tag_id'],
        right_on=['file', 'eventID'], how='left')
    base.loc[~base.eiid.isna(), 'tag_id'] = base.eiid
    base = base[['file', 'sentence', 'tag_id', 'token']]

    # Build tlinks.
    tlinks = get_tags(path, get_tlinks)
    tlinks['eventID'] = tlinks.eventInstanceID.fillna(tlinks.timeID).copy()
    tlinks['relatedTo'] = tlinks.relatedToTime.fillna(tlinks.relatedToEventInstance).copy()
    tlinks.rename(columns={'eventID': 'source'}, inplace=True)
    tlinks = tlinks[['file', 'lid', 'source', 'relatedTo', 'relType']]

    return base, tlinks

