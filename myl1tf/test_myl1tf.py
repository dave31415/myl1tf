import myl1tf
import numpy as np
import cvxopt
import matrix_utils as mu
from l1_everything import l1_fit, l1_fit_monthly
import time
from l1 import l1
from datetime import date
from matplotlib import pylab as plt

doplot = False


def make_l1tf_mock(doplot=doplot, period=6, sea_amp=0.05, noise=0.0):
    np.random.seed(3733)
    num = 100
    x = np.arange(num)
    y = x * 0.0
    y[0:20] = 20.0 + x[0:20] * 1.5
    y[20:50] = y[19] - (x[20:50] - x[19]) * 0.2
    y[50:60] = y[49] + (x[50:60] - x[49]) * 0.47
    y[60:75] = y[59] - (x[60:75] - x[59]) * 2.4
    y[75:] = y[74] + (x[75:] - x[74]) * 2.0
    y = y / y.max()
    y = y + noise * np.random.randn(num)
    if period > 0:
        seas = np.random.randn(period) * sea_amp
        seas_lookup = {k: v for k, v in enumerate(seas)}
        seasonal_part = np.array([seas_lookup[i % period] for i in x])
        seasonal_part = seasonal_part - seasonal_part.mean()
        y_with_seasonal = y + seasonal_part
    else:
        y_with_seasonal = y

    if doplot:
        plt.clf()
        lab ='True, period=%s' % period
        plt.plot(x, y, marker='o', linestyle='-', label=lab, markersize=8, alpha=0.3,color='blue')
        lab = 'True + seasonality, period=%s' % period
        plt.plot(x, y_with_seasonal, marker='o', linestyle='-', label=lab, markersize=8, alpha=0.3,color='red')

    np.random.seed(None)
    return {'x': x, 'y': y, 'y_with_seasonal': y_with_seasonal, 'seas_lookup': seas_lookup}


def make_l1tf_mock2(doplot=doplot, period=6, sea_amp=0.05, noise=0.0, seed=3733):
    np.random.seed(seed)
    num = 100
    x = np.arange(num)
    y = np.zeros(num)
    y[0:20] = 20.0 + x[0:20] * 1.5
    y[20:50] = y[19] - (x[20:50] - x[19]) * 0.2
    y[50:60] = y[49] + (x[50:60] - x[49]) * 0.47
    y[60:75] = y[59] - (x[60:75] - x[59]) * 2.4
    y[75:] = y[74] + (x[75:] - x[74]) * 2.0
    #add two steps
    y = y+ (x <= 30.0)*97.0
    y = y+ (x >85)*87.0
    #add a spike
    y[75] *=16.333
    y=y+x*0.4

    y = y / y.max()



    y = y + noise * np.random.randn(num)

    if period > 0:
        seas = np.random.randn(period) * sea_amp
        seas_lookup = {k: v for k, v in enumerate(seas)}
        seasonal_part = np.array([seas_lookup[i % period] for i in x])
        seasonal_part = seasonal_part - seasonal_part.mean()
        y_with_seasonal = y + seasonal_part
    else:
        y_with_seasonal = y

    if doplot:
        plt.clf()
        lab='True, period=%s' % period
        plt.plot(x, y, marker='o', linestyle='-', label=lab, markersize=8, alpha=0.3,color='blue')
        lab='True + seasonality, period=%s' % period
        plt.plot(x, y_with_seasonal, marker='o', linestyle='-', label=lab, markersize=8, alpha=0.3,color='red')

    np.random.seed(None)
    return {'x': x, 'y': y, 'y_with_seasonal': y_with_seasonal, 'seas_lookup': seas_lookup}


def assert_is_good(resid, mean_abs_resid_max=0.1, max_abs_resid_max =0.15):
    #general form of test on residuals
    mean_resid = resid.mean()
    mean_abs_resid = abs(resid).mean()
    max_abs_resid = abs(resid).max()

    limits = (mean_resid, mean_abs_resid, max_abs_resid)
    print "Mean resid: %s, Mean abs(resid): %s, Max abs(resid): %s" % limits
    assert mean_resid < mean_abs_resid_max
    assert mean_abs_resid < mean_abs_resid_max
    assert max_abs_resid < max_abs_resid_max


def test_l1tf_on_mock(alpha=1.0, beta=0.0, noise=0.0, doplot=doplot):
    if doplot:
        plt.clf()
    mock = make_l1tf_mock(noise=noise)
    l1tf_fit = myl1tf.l1tf(mock['y'], alpha=alpha, beta=beta)

    resid = l1tf_fit['x'] - mock['y_with_seasonal']
    assert_is_good(resid, mean_abs_resid_max=0.035, max_abs_resid_max=0.1)

    if doplot:
        plt.plot(mock['x'], l1tf_fit['x'], marker='o', linestyle='-', label='L1TF, alpha=%s' % alpha)
        plt.legend(loc='lower center')
        plt.show()


def test_l1tf_on_mock_with_period(alpha=1.0, period=6, eta=1.0, doplot=doplot):
    if doplot:
        plt.clf()
    mock = make_l1tf_mock(period=period)
    l1tf_fit = myl1tf.l1tf(mock['y_with_seasonal'], alpha=alpha, period=period, eta=eta)

    resid = l1tf_fit['x'] - mock['y_with_seasonal']
    assert_is_good(resid, mean_abs_resid_max=0.035, max_abs_resid_max=0.1)

    if doplot:
        lab = 'L1TF, period=%s, alpha=%s, eta=%s' % (period, alpha, eta)
        plt.plot(mock['x'], l1tf_fit['x'], marker='o', linestyle='-', markersize=4, alpha=0.8, label=lab)
        lab = 'L1TF + seasonal, period=%s, alpha=%s, eta=%s' % (period, alpha,eta)
        plt.plot(mock['x'], l1tf_fit['x_with_seasonal'], marker='o', markersize=4, linestyle='-', label=lab)
        plt.legend(loc='lower left')
        plt.ylim(0, 1)
        plt.show()


def test_l1tf_on_mock_with_period_l1p(alpha=1.0, period=6, eta=0.1, sea_amp=0.05, doplot=doplot):
    if doplot:
        plt.clf()
    mock = make_l1tf_mock(period=period,sea_amp=sea_amp)
    l1tf_fit = myl1tf.l1tf(mock['y_with_seasonal'], alpha=alpha, period=period, eta=eta, with_l1p=True)

    resid = l1tf_fit['x'] - mock['y_with_seasonal']
    assert_is_good(resid, mean_abs_resid_max=0.035, max_abs_resid_max=0.1)

    if doplot:
        lab = 'L1TF, period=%s, alpha=%s, eta=%s' % (period, alpha, eta)
        plt.plot(mock['x'], l1tf_fit['x'], marker='o', linestyle='-', markersize=4,alpha=0.8,label=lab)
        lab = 'L1TF + seasonal, period=%s, alpha=%s, eta=%s' % (period, alpha,eta)
        plt.plot(mock['x'], l1tf_fit['x_with_seasonal'], marker='o', markersize=4, linestyle='-', label=lab)
        plt.legend(loc='lower left')
        plt.ylim(0, 1)
        plt.show()


def test_l1tf_on_mock_with_period_l1p_with_spike_and_step(alpha=0.5, period=6, eta=0.1,
                    doplot=doplot, sea_amp=0.05):
    #no test for this yet

    if doplot:
        plt.clf()
    mock = make_l1tf_mock2(period=period, sea_amp=sea_amp)
    y = mock['y_with_seasonal']
    ymax=max(y)
    l1tf_fit = myl1tf.l1tf(y, alpha=alpha, period=period, eta=eta, with_l1p=True)
    if doplot:
        lab = 'L1TF, period=%s, alpha=%s, eta=%s' % (period, alpha, eta)
        plt.plot(mock['x'], l1tf_fit['x'], marker='o', linestyle='-', markersize=4,alpha=0.8,label=lab)
        lab = 'L1TF + seasonal, period=%s, alpha=%s, eta=%s' % (period, alpha,eta)
        plt.plot(mock['x'], l1tf_fit['x_with_seasonal'], marker='o', markersize=4, linestyle='-', label=lab)
        plt.legend(loc='upper left')
        plt.ylim(0, ymax)
        plt.plot(mock['x'], y)
        plt.show()
    return l1tf_fit


def test_first_derivative_on_noisy_data(num=10, slope=1.0, offset=3.0, noise=1.0,
                            alpha=1.0, beta=0.0, seed=7863, doplot=doplot):
    if doplot:
        plt.clf()
    np.random.seed(seed)
    i = np.arange(num)
    x = i*slope+offset
    y = x + noise*np.random.randn(num)
    l1tf_fit = myl1tf.l1tf(y, alpha=alpha, beta=beta)

    resid = l1tf_fit['x'] - y
    assert_is_good(resid, mean_abs_resid_max=0.6, max_abs_resid_max=1)

    if doplot:
        plt.plot(i, y, marker='o', markersize=12, alpha=0.8, linestyle='')
        plt.plot(i, l1tf_fit['x'], marker='o', linestyle='-', markersize=4, alpha=0.8)


def test_second_derivative_nes_agrees_with_es():
    #should agree with regularly spaced version when regularly spaced
    n = 13
    D2 = myl1tf.get_second_derivative_matrix(n)
    D2_with_gaps = myl1tf.get_second_derivative_matrix_nes(range(n))
    diff = D2 -D2_with_gaps
    assert max(abs(diff)) < 1e-13


def test_first_derivative_nes_on_quadratic():
    #should agree with regularly spaced version when regularly spaced
    n = 12
    x = np.arange(n)*1.0
    x = x[np.array([1, 2, 5, 9, 11])]
    y = 3.0*x*x + 5.0*x + 99.5
    expected = cvxopt.matrix([6.0*xxx+5.0 for xxx in [2.0, 5.0, 9.0]])
    F = myl1tf.get_first_derivative_matrix_nes(x)
    deriv1 = F*cvxopt.matrix(y)
    diff = deriv1 - expected
    assert max(abs(diff)) < 1e-13


def test_second_derivative_nes_on_quadratic():
    #should agree with regularly spaced version when regularly spaced
    n = 12
    x = np.arange(n)*1.0
    x = x[np.array([1, 2, 5, 9, 11])]
    y = 3.0*x*x + 5.0*x + 99.5
    expected = cvxopt.matrix([6.0 for xxx in [2.0, 5.0, 9.0]])
    F = myl1tf.get_second_derivative_matrix_nes(x)
    deriv2 = F*cvxopt.matrix(y)
    diff = deriv2 - expected
    assert max(abs(diff)) < 1e-13


def test_first_derivative_nes_agrees_with_es():
    #should agree with regularly spaced version when regularly spaced
    n = 13
    F = myl1tf.get_first_derivative_matrix(n)
    F_with_gaps = myl1tf.get_first_derivative_matrix_nes(range(n))
    diff = F -F_with_gaps
    print F
    print F_with_gaps
    max_diff = max(abs(diff))
    print max_diff
    assert max_diff < 1e-13


def test_first_derivative_nes_is_constant_for_line():
    n = 13
    x = np.arange(n)
    F = myl1tf.get_first_derivative_matrix_nes(x)
    xx = cvxopt.matrix(x)
    slope = F*xx
    slope_expected = [1.0]*11
    slope_expected = cvxopt.matrix(slope_expected)
    assert max(abs(slope-slope_expected)) < 1e-13

    #add some gaps, still should be unit slope
    x_with_gaps = x[np.array([1, 4, 5, 9, 11])]
    F = myl1tf.get_first_derivative_matrix_nes(x_with_gaps)
    xx = cvxopt.matrix(x_with_gaps)*3.0+9.5
    slope = F*xx
    slope_expected = [3.0]*(len(x_with_gaps)-2)
    slope_expected = cvxopt.matrix(slope_expected)
    assert max(abs(slope-slope_expected)) < 1e-13


def test_second_derivative_nes_is_zero_for_line():
    n = 13
    x = np.arange(n)
    x_with_gaps = x[np.array([1, 4, 5, 9, 11])]
    D = myl1tf.get_second_derivative_matrix_nes(x_with_gaps)
    slope = D*cvxopt.matrix(x_with_gaps)
    assert max(abs(slope)) < 1e-13


def test_get_B_matrix_nes_aggrees_with_es():
    n = 27
    period = 5
    B_nes = mu.get_B_matrix_nes(np.arange(n), period)
    B_es = mu.get_B_matrix(n, period)

    diff = B_nes-B_es
    max_diff = max(abs(diff))
    assert max_diff < 1e-13


def test_get_B_matrix_nes_on_gap():
    x = np.array([0, 2, 3, 5])
    period = 3
    B_nes = mu.get_B_matrix_nes(x, period)
    expected_matrix = [[1, 0, 0], [0, 0, 1], [1, 0, 0], [0, 0, 1]]
    expected_result = cvxopt.sparse(cvxopt.matrix(expected_matrix).T)
    assert max(B_nes - expected_result) < 1e-13


def test_l1_fit(beta_d2=4.0, beta_d1=1.0, beta_seasonal=1.0, beta_step=2.5, period=12,noise=0, seed=3733):
    mock = make_l1tf_mock2(noise=noise, seed=seed)
    y = mock['y_with_seasonal']
    xx = mock['x']
    plt.clf()
    plt.plot(xx, y, linestyle='-', marker='o', markersize=4)
    sol = l1_fit(xx, y, beta_d2=beta_d2, beta_d1=beta_d1, beta_seasonal=beta_seasonal, beta_step=beta_step, period=period)
    plt.plot(xx, sol['xbase'], label='base')
    plt.plot(xx, sol['steps'], label='steps')
    plt.plot(xx, sol['seas'], label='seasonal')
    plt.plot(xx, sol['x'], label='full')
    plt.legend(loc='upper left')


def test_get_step_function_reg():
    reg = mu.get_step_function_reg(5, 8.0, permissives=None)
    for i in range(5):
        for j in range(5):
            if i != j:
                assert reg[i, j] == 0.0
            else:
                assert reg[i, j] == -8.0

    reg = mu.get_step_function_reg(5, 8.0, permissives=[(2, 7.0), (1, 3.5), (3, 4.5)])
    for i in range(5):
        for j in range(5):
            if i != j:
                assert reg[i, j] == 0.0
            else:
                assert reg[i, j] == [-8.0, -3.5, -7.0, -4.5, -8.0][i]


def test_l1_fit_rand(beta_d2=4.0, beta_d1=1.0, beta_seasonal=1.0, beta_step=2.5,
                     period=12, noise=0, seed=3733,doplot=True, sea_amp=0.05):
    #print "seed=%s,noise=%s,beta_d2=%s,beta_d1=%s,beta_step=%s," \
    #      "beta_seasonal=%s" % (seed,noise,beta_d2,beta_d1,beta_step,beta_seasonal)

    mock = make_l1tf_mock2(noise=noise, seed=seed, sea_amp=sea_amp)
    y = mock['y_with_seasonal']
    xx = mock['x']

    sol = l1_fit(xx, y, beta_d2=beta_d2, beta_d1=beta_d1,
                 beta_seasonal=beta_seasonal, beta_step=beta_step,
                 period=period)

    if doplot:
        plt.clf()
        plt.plot(xx, y, linestyle='-', marker='o', markersize=4)
        plt.plot(xx, sol['xbase'], label='base')
        plt.plot(xx, sol['steps'], label='steps')
        plt.plot(xx, sol['seas'], label='seasonal')
        plt.plot(xx, sol['x'], label='full')
        plt.legend(loc='upper left')


def test_l1_fit_rand_with_permissive(beta_d2=4.0, beta_d1=1.0, beta_seasonal=1.0, beta_step=2.5,
                     period=12, noise=0, seed=3733,doplot=True, sea_amp=0.05):
    #print "seed=%s,noise=%s,beta_d2=%s,beta_d1=%s,beta_step=%s," \
    #      "beta_seasonal=%s" % (seed,noise,beta_d2,beta_d1,beta_step,beta_seasonal)

    mock = make_l1tf_mock2(noise=noise, seed=seed, sea_amp=sea_amp)
    y = mock['y_with_seasonal']
    xx = mock['x']

    step_permissives=[(30, 0.5)]
    sol = l1_fit(xx, y, beta_d2=beta_d2, beta_d1=beta_d1,
                 beta_seasonal=beta_seasonal, beta_step=beta_step,
                 period=period, step_permissives=step_permissives)

    if doplot:
        plt.clf()
        plt.plot(xx, y, linestyle='-', marker='o', markersize=4)
        plt.plot(xx, sol['xbase'], label='base')
        plt.plot(xx, sol['steps'], label='steps')
        plt.plot(xx, sol['seas'], label='seasonal')
        plt.plot(xx, sol['x'], label='full')
        plt.legend(loc='upper left')


def test_l1_fit_speed(beta_d2=4.0, beta_d1=1.0, beta_seasonal=1.0, beta_step=2.5, period=12, n=50, num=100):
    mock = make_l1tf_mock2()
    y = mock['y_with_seasonal']
    xx = mock['x']
    start = time.time()
    for i in xrange(num):
        sol = l1_fit(xx[0:n], y[0:n], beta_d2=beta_d2, beta_d1=beta_d1, beta_seasonal=beta_seasonal, beta_step=beta_step, period=period)
    fin = time.time()
    runtime = fin-start
    rate = num/runtime
    print 'num: %s, runtime: %s seconds, rate: %s per sec for %s points' % (num, runtime, rate, n)
    return runtime


def test_l1_fit_linearity():
    mock = make_l1tf_mock2(noise=0.05, seed=48457)
    y = mock['y_with_seasonal']
    xx = mock['x']

    step_permissives=[(30, 0.5)]
    sol = l1_fit(xx, y, period=6, step_permissives=step_permissives)

    #some linear transformation should effect things only linearly
    scale = 17.0
    offset = -56.0
    y = y*scale + offset

    sol2 = l1_fit(xx, y, period=6, step_permissives=step_permissives)

    x = sol['x']
    x2 = sol2['x']
    x_unscaled = (x2-offset)/scale
    tol = 1e-10

    assert abs(x-x_unscaled).max() < tol


def test_l1_fit_step():
    #test that the place of the step and the specification of the step
    #permissive do not have a off-by-one error
    #note our steps occur from right to left
    #the first one (right to left) that has jumped is the place
    #where h is non-zero
    n = 55
    xx = np.arange(n)
    y = xx*0.0
    y[xx <= 20] = 1000.0
    #this means that y[20] includes the step-up and so 20 is where 'h' should be non-zero
    sol = l1_fit(xx, y, beta_step=100.0, step_permissives=[(20, 0.1)])
    print abs(sol['h'][19]) < 1e-10
    print abs(sol['h'][21]) < 1e-10
    print abs(sol['h'][20]-1000.0) < 1e-10


def test_l1():
    np.random.seed(42)
    m, n = 500, 100
    P, q = cvxopt.normal(m, n), cvxopt.normal(m, 1)
    u = l1(P, q)
    qfit = P*u
    residual = qfit-q
    np.random.seed(None)
    mean_abs_res = sum(abs(residual))/len(residual)
    print 'mean abs residual: %s' % mean_abs_res
    assert mean_abs_res < 1.0


def test_gaps_work_on_line():
    xx = np.arange(20, dtype=np.int64)
    y = 3 * xx + 9
    sol = l1_fit(xx, y)
    xfit = sol['x']
    #this one has a big gap
    xx = np.array(range(7)+range(15, 20))
    y = 3 * xx + 9
    sol = l1_fit(xx, y)
    xfit2 = sol['x']
    for i, j in enumerate(xx):
        diff = abs(xfit[j] - xfit2[i])
        assert diff < 1e-6


def test_same_result_with_offset_index():
    xx = np.arange(20, dtype=np.int64)
    y = 3 * xx + 9
    sol1 = l1_fit(xx, y)
    xfit1 = sol1['x']
    xx= xx + 999
    sol2 = l1_fit(xx, y)
    xfit2 = sol2['x']
    assert (xfit1 == xfit2).all()


def test_dates_to_index_monthly():
    dates = [date(2000, 1, 1), date(2000, 2, 27), date(2014, 1, 22),
             date(1999, 12, 1), date(1998, 2, 15)]
    index = mu.dates_to_index_monthly(dates)
    print index
    assert (index == np.array([0, 1, 12*14, -1, -23])).all()


def test_l1_fit_monthly():
    tol = 1e-12
    xx = np.arange(4, dtype=np.int64)
    y = 3 * xx + 9
    sol1 = l1_fit(xx, y)
    xfit1 = sol1['x']
    xx = np.array([date(2014, 1, 2), date(2014, 2, 5), date(2014, 3, 17), date(2014, 4 ,1)])
    sol2 = l1_fit_monthly(xx, y)
    xfit2 = sol2['x']
    diff = abs(xfit1 - xfit2).max()
    assert diff < tol

