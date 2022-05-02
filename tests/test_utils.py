from tieval.utils import get_spans


def test_get_spans():

    text = 'Pacific First Financial Corp. said shareholders approved its acquisition by Royal Trustco Ltd. of Toronto ' \
           'for $27 a share, or $212 million.\nThe thrift holding company said it expects to obtain regulatory ' \
           'approval and complete the transaction by year-end. '
    tokens = ['Pacific', 'First', 'Financial', 'Corp.', 'said', 'shareholders', 'approved', 'its', 'acquisition', 'by',
              'Royal', 'Trustco', 'Ltd.', 'of', 'Toronto', 'for', '$', '27', 'a', 'share', ',', 'or', '$', '212',
              'million', '.', 'The', 'thrift', 'holding', 'company', 'said', 'it', 'expects', 'to', 'obtain',
              'regulatory', 'approval', 'and', 'complete', 'the', 'transaction', 'by', 'year-end', '.']

    spans = get_spans(text, tokens)
    for span, tkn in zip(spans, tokens):
        s, e = span
        assert text[s: e] == tkn
