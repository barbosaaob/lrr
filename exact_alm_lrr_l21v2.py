import numpy as np
from solve_l1l2 import solve_l1l2


def exact_alm_lrr_l21v2(D, A, lamb, tol=1e-7, maxIter=1000, display=False):
    m, n = D.shape
    k = A.shape[1]

    maxIter_primal = 10000
    # initialize
    Y = np.sign(D)
    norm_two = np.linalg.norm(Y, 2)
    norm_inf = np.linalg.norm(Y.flatten(1), np.inf) / lamb
    dual_norm = max(norm_two, norm_inf)
    Y /= dual_norm

    W = np.zeros((k, n))

    Z_hat = np.zeros((k, n))
    E_hat = np.zeros((m, n))
    # parameters
    dnorm = np.linalg.norm(D, 'fro')
    tolProj1 = 1e-6 * dnorm

    anorm = np.linalg.norm(A, 2)
    tolProj2 = 1e-6 * dnorm / anorm

    mu = 0.5 / norm_two  # this one can be tuned
    rho = 6              # this one can be tuned

    # pre-computation
    if m >= k:
        inv_ata = np.linalg.inv(np.eye(k) + A.T.dot(A))
    else:
        inv_ata = np.eye(k) - np.linalg.solve((np.eye(m) + A.dot(A.T)).T,
                                              A).T.dot(A)

    iter = 0
    while iter < maxIter:
        iter += 1

        # solve the primal problem by alternative projection
        primal_iter = 0

        while primal_iter < maxIter_primal:
            primal_iter += 1
            temp_Z, temp_E = Z_hat, E_hat

            # update J
            temp = temp_Z + W / mu
            U, S, V = np.linalg.svd(temp, 'econ')
            V = V.T

            diagS = S
            svp = len(np.flatnonzero(diagS > 1.0 / mu))
            diagS = np.maximum(0, diagS - 1.0 / mu)

            if svp < 0.5:  # svp = 0
                svp = 1

            J_hat = U[:, 0:svp].dot(np.diag(diagS[0:svp]).dot(V[:, 0:svp].T))

            # update Z
            temp = J_hat + A.T.dot(D - temp_E) + (A.T.dot(Y) - W) / mu
            Z_hat = inv_ata.dot(temp)

            # update E
            temp = D - A.dot(Z_hat) + Y / mu
            E_hat = solve_l1l2(temp, lamb / mu)

            if np.linalg.norm(E_hat - temp_E, 'fro') < tolProj1 and \
               np.linalg.norm(Z_hat - temp_Z) < tolProj2:
                break

        H1 = D - A.dot(Z_hat) - E_hat
        H2 = Z_hat - J_hat
        Y = Y + mu * H1
        W = W + mu * H2
        mu = rho * mu

        # stop Criterion
        stopCriterion = max(np.linalg.norm(H1, 'fro') / dnorm,
                            np.linalg.norm(H2, 'fro') / dnorm * anorm)
        if display:
            print 'LRR: Iteration', iter, '(', primal_iter, '), mu ', mu, \
                  ', |E|_2,0 ', np.sum(np.sum(E_hat ** 2, 1) > 0), \
                  ', stopCriterion ', stopCriterion

        if stopCriterion < tol:
            break

    return (Z_hat, E_hat)
