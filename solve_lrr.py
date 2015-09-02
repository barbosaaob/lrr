import numpy as np
from scipy.linalg import orth
from exact_alm_lrr_l21v2 import exact_alm_lrr_l21v2
from exact_alm_lrr_l1v2 import exact_alm_lrr_l1v2
# import inexact_alm_lrr_l21
# import inexact_alm_lrr_l1


def solve_lrr(X, A, lamb, reg=0, alm_type=0, display=False):
    Q = orth(A.T)
    B = A.dot(Q)

    if reg == 0:
        if alm_type == 0:
            Z, E = exact_alm_lrr_l21v2(X, B, lamb, 1e-7, 1000, display)
        # else:
        #     Z, E = inexact_alm_lrr_l21(X, B, lamb, display)
    else:
        if alm_type == 0:
            Z, E = exact_alm_lrr_l1v2(X, B, lamb, 1e-7, 1000, display)
        # else:
        #     Z, E = inexact_alm_lrr_l1(X, B, lamb, display)

    Z = Q.dot(Z)
    return (Z, E)


if __name__ == "__main__":
    data = np.random.random((10, 3))
    X = data.T
    A = X
    lamb = 0.1
    Z, E = solve_lrr(X, A, lamb)
    print "Z: ", Z
    print "E: ", E
