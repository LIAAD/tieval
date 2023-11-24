from tieval.evaluate import confusion_matrix, Offsets


def test_confusion_matrix():

    pred = {1, 2, 3}
    true = {3, 4}

    tp, fp, fn = confusion_matrix(true, pred)

    assert tp == 1
    assert fp == 2
    assert fn == 1


def test_confusion_matrix_span():

    pred = [Offsets(10, 20)]
    true = [Offsets(5, 15)]
    tp, fp, fn = confusion_matrix(true, pred)
    assert tp == 1
    assert fp == 0
    assert fn == 0

    pred = [Offsets(10, 20)]
    true = [Offsets(15, 25)]
    tp, fp, fn = confusion_matrix(true, pred)
    assert tp == 1
    assert fp == 0
    assert fn == 0

    pred = [Offsets(10, 20)]
    true = [Offsets(20, 30)]
    tp, fp, fn = confusion_matrix(true, pred)
    assert tp == 1
    assert fp == 0
    assert fn == 0

    pred = [Offsets(10, 20)]
    true = [Offsets(5, 10)]
    tp, fp, fn = confusion_matrix(true, pred)
    assert tp == 1
    assert fp == 0
    assert fn == 0

    pred = [Offsets(10, 20)]
    true = [Offsets(1, 5)]
    tp, fp, fn = confusion_matrix(true, pred)
    assert tp == 0
    assert fp == 1
    assert fn == 1
