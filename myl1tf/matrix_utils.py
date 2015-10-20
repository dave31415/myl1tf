from itertools import chain
from cvxopt import spmatrix, sparse, matrix
import numpy as np


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


def get_first_derivative_matrix(n):
    """
    :param x:  numpy array of x-values
    :return: A matrix D such that if x.size == (n,1), D * x is the first derivative of x
    :symmetric first derivative (excludes endpoints like as before)
    """
    m = n - 2
    F = spmatrix(list(chain(*[[-0.5, 0.0, 0.5]] * m)),
                 list(chain(*[[i] * 3 for i in xrange(m)])),
                 list(chain(*[[i, i + 1, i + 2] for i in xrange(m)])))
    return F


def get_second_derivative_matrix_nes(x):
    """
    Get the second derivative matrix for non-equally spaced points
    :param : x numpy array of x-values
    :return: A matrix D such that if x.size == (n,1), D * x is the second derivative of x
    assumes points are sorted
    """
    n = len(x)
    m = n - 2

    values = []
    for i in xrange(1, n-1):
        a0 = float(x[i+1] - x[i])
        a1 = float(x[i+1] - x[i-1])
        a2 = float(x[i] - x[i-1])
        a = a0 * a1 * a2
        assert (a0 >= 0) and (a1 >= 0) and (a2 >= 0), "Points do not appear to be sorted"
        assert a != 0, "Second derivative doesn't exist for repeated points"
        vals = [2.0/(a1*a2), -2.0/(a0*a2), 2.0/(a0*a1)]
        values.extend(vals)

    D = spmatrix(values,
                 list(chain(*[[i] * 3 for i in xrange(m)])),
                 list(chain(*[[i, i + 1, i + 2] for i in xrange(m)])))
    return D


def get_step_function_matrix(n):
    """
    Upper/lower triangular with all ones
    :param n:
    :return:
    """

    step = identity_spmatrix(n)
    for i in xrange(n):
        for j in xrange(n):
            if i < j:
                step[i, j] = 1.0
    return step


def get_first_derivative_matrix_nes(x):
    """
    Get the first derivative matrix for non-equally spaced points
    :param n: The size of the time series
    :return: A matrix D such that if x.size == (n,1), D * x is the second derivative of x
    assumes points are sorted
    """
    n = len(x)
    m = n - 2

    values = []
    for i in xrange(1, n-1):
        a0 = float(x[i+1] - x[i])
        a1 = float(x[i+1] - x[i-1])
        a2 = float(x[i] - x[i-1])
        a = a0 * a1 * a2
        assert (a0 >= 0) and (a1 >= 0) and (a2 >= 0), "Points do not appear to be sorted"
        assert a != 0, "Second derivative doesn't exist for repeated points"
        vals = [(x[i]-x[i+1])/(a1*a2), (x[i-1]+x[i+1]-2.0*x[i])/(a0*a2), (x[i]-x[i-1])/(a0*a1)]
        values.extend(vals)

    D = spmatrix(values,
                 list(chain(*[[i] * 3 for i in xrange(m)])),
                 list(chain(*[[i, i + 1, i + 2] for i in xrange(m)])))
    return D


def spmatrix2np(spmat):
    """
        Convert a matrix or spmatrix to numpy 2D array
    :param spmat: matrix or spmatrix
    :return: numpy 2D array of type float64
    """
    return np.asarray(matrix(spmat)).squeeze()


def np2spmatrix(nparray):
    """
        Convert a numpy ndarray to sparse cvxopt matrix
    :param nparray: numpy ndarray
    :return: cvxopt sparse matrix
    """
    return sparse(matrix(nparray))


def scale_numpy(y):
    """
    :param y: a numpy ndarray
    :return: scaled to [0,1] range
    """
    assert isinstance(y, np.ndarray)
    y_min = float(y.min())
    y_max = float(y.max())
    denom = y_max - y_min
    # if denom == 0, y is constant
    y_scaled = (y - y_min) / (1 if denom == 0 else denom)
    return y_scaled, y_min, y_max


def unscale_numpy(y_scaled, y_min, y_max):
    """
    :param y_scaled: a numpy ndarray
    :param y_min: min of y, output by scale_numpy
    :param y_max: max of y, output by scale_numpy
    :return: scaled numpy array to original range
    """
    if y_max > y_min:
        y = y_scaled * (y_max-y_min)
    else:
        y = y_scaled
    y = y + y_min
    return y


def invert(spmat):
    """
    :param spmat: a cvx sparse matrix
    :return: the inverse matrix as sparse
    """
    arr = spmatrix2np(spmat)
    arr_inv = np.linalg.inv(arr)
    return sparse(matrix(arr_inv))


def pinvert(spmat):
    """
    :param spmat: a cvx sparse matrix
    :return: the pseudo-inverse matrix as sparse
    """
    arr = spmatrix2np(spmat)
    arr_inv = np.linalg.pinv(arr)
    return sparse(matrix(arr_inv))


def get_B_matrix(n, period):
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
    B = B[0:n, :]
    return B


def get_B_matrix_nes(x, period):
    """
    :param x: numpy arrays of type integer
    :param period: length of period
    :return: B matrix which maps p -> S cyclically
    """
    x_min = min(x)
    nx = len(x)
    index = (x-x_min) % period
    B = zero_spmatrix(nx, period)
    for row, i in enumerate(index):
        B[row, i] = 1.0
    return B


def get_T_matrix(period):
    T = zero_spmatrix(period, m=period-1)
    T[0:period-1, :] = identity_spmatrix(period-1)
    T[period-1, :] = -1.0
    return T


def identity_spmatrix(n):
    return spmatrix(1.0, range(n), range(n))


def zero_spmatrix(n, m=None):
    if m is None:
        m = n
    return spmatrix(0.0, [n-1], [m-1])


def get_step_function_reg(n, beta_step, permissives=None):
    #step function regularization matrix
    reg = -beta_step*identity_spmatrix(n)
    if permissives is not None:
        #these points may have more permissive regularization values
        #for example where you expect jumps to be more natural
        #such as at obvious boundaries
        for point in permissives:
            i, beta = point
            reg[i, i] = -beta
    return reg


def date_to_index_monthly(date_in):
    #months since Jan 2000, ignores the day
    return 12*(date_in.year-2000) + (date_in.month-1)


def dates_to_index_monthly(dates):
    #months since Jan 2000, ignores the day
    return np.array([date_to_index_monthly(d) for d in dates])









