# Adapted from py-l1tf here https://github.com/elsonidoq/py-l1tf/
from cvxopt import solvers, matrix
import l1
import numpy as np
solvers.options['show_progress'] = 0
from matrix_utils import *


def l1tf(y, alpha, period=0, eta=1.0, with_l1p=False, beta=0.0):
    # scaling things to standardized size
    y_min = float(y.min())
    y_max = float(y.max())
    denom = y_max - y_min
    # if denom == 0, y is constant
    y_scaled = (y - y_min) / (1 if denom == 0 else denom)

    assert isinstance(y, np.ndarray)
    if with_l1p:
        solution = l1tf_cvxopt_l1p(matrix(y_scaled), alpha, period=period, eta=eta)
    else:
        solution = l1tf_cvxopt(matrix(y_scaled), alpha, period=period, eta=eta, beta=beta)

    #convert back to unscaled, numpy arrays
    for k, v in solution.iteritems():
        #don't add the baseline to seasonal parts which should have zero mean
        add_base_line = k not in ['p', 's']
        solution[k] = np.asarray(v * (y_max - y_min) + y_min*add_base_line).squeeze()
    return solution


def l1tf_cvxopt(y, alpha, period=0, eta=1.0, beta=0.0):
    n = y.size[0]
    m = n - 2

    D = get_second_derivative_matrix(n)
    if beta > 0:
        #put a penalty on the l1 norm of the first dervative as well
        F = get_first_derivative_matrix(n)
        D_F = zero_spmatrix(2*m, n)
        D_F[:m, :n] = D
        D_F[m:, :n] = F * beta
        D = D_F
        m *= 2

    P = D * D.T
    if period > 0:
        B = get_B_matrix(n, period)
        T = get_T_matrix(period)
        Q = B*T
        DQ = D * Q
        TT = T.T * T
        TTI = invert(TT)
        P_seasonal = (1.0/eta) * DQ * TTI * DQ.T
        P += P_seasonal

    q = -D * y

    G = zero_spmatrix(2*m, m)
    G[:m, :m] = identity_spmatrix(m)
    G[m:, :m] = - identity_spmatrix(m)

    h = matrix(alpha, (2 * m, 1), tc='d')

    res = solvers.qp(P, q, G, h)

    nu = res['x']
    DT_nu = D.T * nu

    output={}
    output['y'] = y
    output['x_with_seasonal'] = y - DT_nu
    output['x'] = y - DT_nu
    if period > 0:
        output['p'] = (1.0/eta) * TTI * Q.T * DT_nu
        output['s'] = Q * output['p']
        output['x'] -= output['s']
        print 'sum seasonal: %s' % sum(output['s'][:period])
    return output


def l1tf_cvxopt_l1p(y, alpha, period=0, eta=1.0):
    n = y.size[0]
    m = n - 2

    D = get_second_derivative_matrix(n)

    P = D * D.T

    q = -D * y

    n_contraints = m
    if period > 1:
        n_contraints += (period-1)

    G = zero_spmatrix(2 * n_contraints, m)
    G[:m, :m] = identity_spmatrix(m)
    G[m:2*m, :m] = - identity_spmatrix(m)
    h = matrix(alpha, (2 * n_contraints, 1), tc='d')

    if period > 1:
        B = get_B_matrix(n, period)
        T = get_T_matrix(period)
        Q = B*T
        DQ = D * Q
        G[2*m:2*m+period-1, :m] = DQ.T
        G[2*m+period-1:, :m] = -DQ.T
        h[2*m:] = eta

    res = solvers.qp(P, q, G, h)

    nu = res['x']
    DT_nu = D.T * nu

    output = {}
    output['y'] = y
    output['x_with_seasonal'] = y - DT_nu
    if period > 1:
        #separate seasonal from non-seasonal by solving an
        #least norm problem
        ratio= eta/alpha
        Pmat = zero_spmatrix(m+period, period-1)
        Pmat[:m, :period-1] = DQ
        Pmat[m:(m+period), :period-1] = -ratio * T
        qvec = matrix(0.0, (m+period, 1), tc='d')
        qvec[:m] = D*(y-DT_nu)
        p_solution = l1.l1(matrix(Pmat), qvec)
        QP_solution = Q*p_solution
        output['p'] = p_solution
        output['s'] = QP_solution
        output['x'] = output['x_with_seasonal'] - output['s']
        print 'sum seasonal is: %s' % sum(output['s'][:period])

    return output




