# Adapted from py-l1tf here https://github.com/elsonidoq/py-l1tf/
from itertools import chain
from cvxopt import matrix, spmatrix, solvers,sparse
import numpy as np
solvers.options['show_progress'] = 0


def get_second_derivative_matrix(n):
    """
    :param n: The size of the time series

    :return: A matrix D such that if x.size == (n,1), D * x is the second derivative of x
    """
    m = n - 2

    D = spmatrix(list(chain(*[[1, -2, 1]] * m)),
                 list(chain(*[[i] * 3 for i in xrange(m)])),
                 list(chain(*[[i, i + 1, i + 2] for i in xrange(m)])))
    return D

def l1tf(y, alpha, primary=False,period=0,psi=1.0):
    # scaling things to standardized size
    y_min = float(y.min())
    y_max = float(y.max())
    denom = y_max - y_min
    # if denom == 0, y is constant
    y_scaled = (y - y_min) / (1 if denom == 0 else denom)

    assert isinstance(y, np.ndarray)
    solution = l1tf_cvxopt(matrix(y_scaled), alpha, period=period, psi=psi)
    #convert back to unscaled, numpy arrays
    for k, v in solution.iteritems():
        solution[k] = np.asarray(v * (y_max - y_min) + y_min).squeeze()
    return solution


def l1tf_cvxopt(y, alpha, period=0, psi=1.0):
    n = y.size[0]
    m = n - 2

    D = get_second_derivative_matrix(n)

    P = D * D.T
    if period > 0:
        B = B_matrix(n, period)
        T = zero_spmatrix(period, m=period-1)
        T[0:period-1, :] = identity_spmatrix(period-1)
        T[period-1, :] = -1.0
        Q=B*T
        G = D * Q
        P_seasonal = (1.0/psi) * G * G.T
        P += P_seasonal

    q = -D * y

    G = spmatrix([], [], [], (2 * m, m))
    G[:m, :m] = identity_spmatrix(m)
    G[m:, :m] = - identity_spmatrix(m)

    h = matrix(alpha, (2 * m, 1), tc='d')

    res = solvers.qp(P, q, G, h)

    nu = res['x']
    DT_nu = D.T * nu

    output={}
    output['y'] = y
    output['nu'] = nu
    output['x'] = y - DT_nu
    if period > 0:
        output['x'] -= (1.0/psi) * Q * Q.T * DT_nu
        output['p'] = (1.0/psi) * Q.T * DT_nu
        output['s'] = Q * output['p']
        output['x_with_seasonal'] = output['x'] + output['s']
    return output


def spmatrix2np(spmat):
    """
        Convert a matrix or spmatrix to numpy 2D array
    :param spmat: matrix or spmatrix
    :return: numpy 2D array of type float64
    """
    return np.asarray(matrix(spmat))

def invert(spmat):
    """
    :param spmat: a cvx sparse matrix
    :return: the inverse matrix as sparse
    """
    arr = spmatrix2np(spmat)
    arr_inv = np.linalg.inv(arr)
    return sparse(matrix(arr_inv))

def B_matrix(n, period):
    """
    :param n: number of target variables
    :param period: length of period
    :return: B matrix which maps p -> S cyclically
    S_i = p_j (where j = i mod period)
    """
    num_full_cycles = int(np.ceil(n/float(period)))
    identity_p = identity_spmatrix(period)
    num = num_full_cycles * period
    B = zero_spmatrix(num, period)
    for i in xrange(num_full_cycles):
        B[i*period:(i+1)*period, :] = identity_p
    #trim off excess
    B=B[0:n, :]
    return B


def identity_spmatrix(n):
    return spmatrix(1.0, range(n), range(n))


def zero_spmatrix(n,m=None):
    if m is None:
        m = n
    return spmatrix(0.0, [n-1], [m-1])




