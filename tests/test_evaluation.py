import numpy as np

from ngboost.evaluation import calculate_concordance_naive


def _old_calculate_concordance_naive(preds, Y, E):
    trues = Y
    concordance, N = 0, len(trues)
    counter = 0
    for i in range(N):
        for j in range(i + 1, N):
            cond_1 = E[i] and E[j]
            cond_2 = E[i] and not E[j] and Y[i] < Y[j]
            cond_3 = not E[i] and E[j] and Y[i] > Y[j]
            if cond_1 or cond_2 or cond_3:
                if (preds[i] < preds[j] and trues[i] < trues[j]) or (
                    preds[i] > preds[j] and trues[i] > trues[j]
                ):
                    concordance += 1
                elif preds[i] == preds[j]:
                    concordance += 0.5
                counter += 1
    return concordance / counter


def test_calculate_concordance_naive_matches_pairwise_implementation():
    preds = np.array([0.2, 0.7, 0.4, 0.9, 0.1, 0.8])
    Y = np.array([1.0, 4.0, 2.0, 6.0, 3.0, 5.0])
    E = np.array([True, True, False, True, False, True])

    assert calculate_concordance_naive(preds, Y, E) == (
        _old_calculate_concordance_naive(preds, Y, E)
    )


def test_calculate_concordance_naive_preserves_uncensored_time_ties():
    preds = np.array([0.2, 0.7, 0.4, 0.9, 0.4])
    Y = np.array([1.0, 1.0, 2.0, 3.0, 2.0])
    E = np.array([True, True, True, True, True])

    assert calculate_concordance_naive(preds, Y, E) == (
        _old_calculate_concordance_naive(preds, Y, E)
    )


def test_calculate_concordance_naive_preserves_same_time_censoring():
    preds = np.array([0.2, 0.7, 0.4, 0.9, 0.4, 0.8])
    Y = np.array([1.0, 1.0, 2.0, 3.0, 2.0, 3.0])
    E = np.array([True, False, True, True, False, False])

    assert calculate_concordance_naive(preds, Y, E) == (
        _old_calculate_concordance_naive(preds, Y, E)
    )


def test_calculate_concordance_naive_accepts_list_inputs():
    preds = [0.2, 0.7, 0.4, 0.9, 0.4, 0.8]
    Y = [1.0, 1.0, 2.0, 3.0, 2.0, 3.0]
    E = [True, False, True, True, False, False]

    assert calculate_concordance_naive(preds, Y, E) == (
        _old_calculate_concordance_naive(preds, Y, E)
    )
