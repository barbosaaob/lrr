import numpy as np
import sys
sys.path.insert(0, '/home/adriano/workspace/lrr')

from solve_lrr import solve_lrr

data = np.loadtxt('/tmp/data')
X = data.T
A = X
l = 0.1

Z, E = solve_lrr(X, A, l, reg=1, alm_type=1, display=True)

np.savetxt('/tmp/Z', Z)
np.savetxt('/tmp/E', E)
