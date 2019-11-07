import numpy as np
import time
from solve_lrr import solve_lrr


def demo():
    A = np.random.randn(100, 200)
    X = np.random.randn(100, 100)
    lamb = 0.01

    print("solve min |Z|_* + lambda |E|_21 s.t. X = AZ + E by exact ALM ...")

    tic = time.time()
    Z1, E1 = solve_lrr(X, A, lamb, reg=0, alm_type=0)
    obj1 = np.sum(np.linalg.svd(Z1)[1]) + lamb * np.sum(np.sqrt(np.sum(E1 ** 2,
                                                                       1)))
    print("Elapsed time:", time.time() - tic)

    print("objective value=", obj1)

    print("solve min |Z|_* + lambda |E|_21 s.t. X = AZ + E by inexact ALM ...")

    tic = time.time()
    Z2, E2 = solve_lrr(X, A, lamb, reg=0, alm_type=1)
    obj2 = np.sum(np.linalg.svd(Z2)[1]) + lamb * np.sum(np.sqrt(np.sum(E2 ** 2,
                                                                       1)))
    print("Elapsed time:", time.time() - tic)
    print("objective value=", obj2)

    diff = np.max(np.abs(Z1 - Z2))

    print("### Warning: difference of the solution found by those two \
          approaches: |Z1 - Z2|_inf=%f" % diff)

    print("solve min |Z|_* + lambda |E|_1 s.t. X = AZ + E by exact ALM ...")
    tic = time.time()
    Z1, E1 = solve_lrr(X, A, lamb, reg=1, alm_type=0)
    obj1 = np.sum(np.linalg.svd(Z1)[1]) + lamb * np.sum(np.sqrt(np.sum(E1 ** 2,
                                                                       1)))
    print("Elapsed time:", time.time() - tic)

    print("objective value=", obj1)

    print("solve min |Z|_* + lambda |E|_1 s.t. X = AZ + E by inexact ALM ...")
    tic = time.time()
    Z2, E2 = solve_lrr(X, A, lamb, reg=1, alm_type=1)
    obj2 = np.sum(np.linalg.svd(Z2)[1]) + lamb * np.sum(np.sqrt(np.sum(E2 ** 2,
                                                                       1)))
    print("Elapsed time:", time.time() - tic)
    print("objective value=", obj2)

    diff = np.max(np.abs(Z1 - Z2))

    print("### Warning: difference of the solution found by those two\
          approaches: |Z1 - Z2|_inf=", diff)


if __name__ == "__main__":
    demo()
