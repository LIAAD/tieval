from tieval.evaluate import confusion_matrix, Span


def test_confusion_matrix():

    pred = {1, 2, 3}
    true = {3, 4}

    tp, fp, fn = confusion_matrix(true, pred)

    assert tp == 1
    assert fp == 2
    assert fn == 1


def test_confusion_matrix_span():

    pred = [Span(10, 20)]
    true = [Span(5, 15)]
    tp, fp, fn = confusion_matrix(true, pred)
    assert tp == 1
    assert fp == 0
    assert fn == 0

    pred = [Span(10, 20)]
    true = [Span(15, 25)]
    tp, fp, fn = confusion_matrix(true, pred)
    assert tp == 1
    assert fp == 0
    assert fn == 0

    pred = [Span(10, 20)]
    true = [Span(20, 30)]
    tp, fp, fn = confusion_matrix(true, pred)
    assert tp == 1
    assert fp == 0
    assert fn == 0

    pred = [Span(10, 20)]
    true = [Span(5, 10)]
    tp, fp, fn = confusion_matrix(true, pred)
    assert tp == 1
    assert fp == 0
    assert fn == 0

    pred = [Span(10, 20)]
    true = [Span(1, 5)]
    tp, fp, fn = confusion_matrix(true, pred)
    assert tp == 0
    assert fp == 1
    assert fn == 1


