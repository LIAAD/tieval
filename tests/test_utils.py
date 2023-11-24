from tieval.utils import (
    get_offsets,
    resolve_sentence_idxs
)


def test_get_offsets():
    text = 'Pacific First Financial Corp. said shareholders approved its acquisition by Royal Trustco Ltd. of ' \
           'Toronto for $27 a share, or $212 million.\nThe thrift holding company said it expects to obtain ' \
           'regulatory approval and complete the transaction by year-end. '
    tokens = ['Pacific', 'First', 'Financial', 'Corp.', 'said', 'shareholders', 'approved', 'its', 'acquisition', 'by',
              'Royal', 'Trustco', 'Ltd.', 'of', 'Toronto', 'for', '$', '27', 'a', 'share', ',', 'or', '$', '212',
              'million', '.', 'The', 'thrift', 'holding', 'company', 'said', 'it', 'expects', 'to', 'obtain',
              'regulatory', 'approval', 'and', 'complete', 'the', 'transaction', 'by', 'year-end', '.']

    offsets = get_offsets(text, tokens)
    for offsets, tkn in zip(offsets, tokens):
        s, e = offsets
        assert text[s: e] == tkn


def test_resolve_sentence_idxs():
    assert resolve_sentence_idxs(None, 2) == [2]
    assert resolve_sentence_idxs(1, None) == [1]
    assert resolve_sentence_idxs(1, 1) == [1]
    assert resolve_sentence_idxs(1, 2) == [1, 2]
    assert resolve_sentence_idxs(2, 1) == [1, 2]
