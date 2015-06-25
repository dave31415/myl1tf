import matrix_utils as mu
import l1
import numpy as np
from cvxopt import spmatrix, matrix, sparse


def l1_fit_monthly(dates, y, **kwargs):
    #wrapper that allows an numpy array of dates
    #that will be converted to an index
    #defined as months since Jan 2000. days are ignored
    xx = mu.dates_to_index_monthly(dates)
    return l1_fit(xx, y, **kwargs)


def l1_fit(index, y, beta_d2=1.0, beta_d1=1.0, beta_seasonal=1.0,
           beta_step=5.0, period=12, growth=0.0, step_permissives=None):
    assert isinstance(y, np.ndarray)
    assert isinstance(index, np.ndarray)
    #x must be integer type for seasonality to make sense
    assert index.dtype.kind == 'i'
    n = len(y)
    m = n-2
    p = period

    ys, y_min, y_max = mu.scale_numpy(y)

    D1 = mu.get_first_derivative_matrix_nes(index)
    D2 = mu.get_second_derivative_matrix_nes(index)
    H = mu.get_step_function_matrix(n)
    T = mu.get_T_matrix(p)
    B = mu.get_B_matrix_nes(index, p)
    Q = B*T

    #define F_matrix from blocks like in paper
    zero = mu.zero_spmatrix
    ident = mu.identity_spmatrix
    gvec = spmatrix(growth, range(m), [0]*m)
    zero_m = spmatrix(0.0, range(m), [0]*m)
    zero_p = spmatrix(0.0, range(p), [0]*p)
    zero_n = spmatrix(0.0, range(n), [0]*n)

    step_reg = mu.get_step_function_reg(n, beta_step, permissives=step_permissives)

    F_matrix = sparse([
        [ident(n), -beta_d1*D1, -beta_d2*D2, zero(p, n), zero(n)],
        [Q, zero(m, p-1), zero(m, p-1), -beta_seasonal*T, zero(n, p-1)],
        [H, zero(m, n), zero(m, n), zero(p, n), step_reg]
    ])

    w_vector = sparse([
        mu.np2spmatrix(ys), gvec, zero_m, zero_p, zero_n
    ])

    solution_vector = np.asarray(l1.l1(matrix(F_matrix), matrix(w_vector))).squeeze()
    #separate
    xbase = solution_vector[0:n]
    s = solution_vector[n:n+p-1]
    h = solution_vector[n+p-1:]
    #scale back to original
    if y_max > y_min:
        scaling = y_max - y_min
    else:
        scaling = 1.0

    xbase = xbase*scaling + y_min
    s = s*scaling
    h = h*scaling
    seas = np.asarray(Q*matrix(s)).squeeze()
    steps = np.asarray(H*matrix(h)).squeeze()
    x = xbase + seas + steps

    solution = {'xbase': xbase, 'seas': seas, 'steps': steps, 'x': x, 'h': h, 's': s}
    return solution

