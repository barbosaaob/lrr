import numpy as np


def solve_l1l2(W, lamb):
    n = W.shape[1]
    E = W.copy()
    for i in range(n):
        E[:, i] = solve_l2(W[:, i], lamb)
    return E


def solve_l2(w, lamb):
    nw = np.linalg.norm(w)
    if nw > lamb:
        x = (nw - lamb) * w / float(nw)
    else:
        x = np.zeros((max(w.shape),))
    return x
