from tieval.evaluate import confusion_matrix


def test_confusion_matrix():

    pred = {1, 2, 3}
    true = {3, 4}

    tp, fp, fn = confusion_matrix(true, pred)

    assert tp == 1
    assert fp == 2
    assert fn == 1


