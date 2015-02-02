import myl1tf
import numpy as np
from matplotlib import pylab as plt
import cvxopt

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
        lab='True, period=%s' % period
        plt.plot(x, y, marker='o', linestyle='-', label=lab, markersize=8, alpha=0.3,color='blue')
        lab='True + seasonality, period=%s' % period
        plt.plot(x, y_with_seasonal, marker='o', linestyle='-', label=lab, markersize=8, alpha=0.3,color='red')

    return {'x': x, 'y': y, 'y_with_seasonal': y_with_seasonal, 'seas_lookup': seas_lookup}

def assert_is_good(resid, mean_resid_max=1e-9, mean_abs_resid_max=0.1, max_abs_resid_max =0.15):
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
    assert_is_good(resid, mean_resid_max=1e-12, mean_abs_resid_max=0.035, max_abs_resid_max=0.1)

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
    assert_is_good(resid, mean_resid_max=6e-4, mean_abs_resid_max=0.035, max_abs_resid_max=0.1)

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
    assert_is_good(resid, mean_resid_max=6e-4, mean_abs_resid_max=0.035, max_abs_resid_max=0.1)

    if doplot:
        lab = 'L1TF, period=%s, alpha=%s, eta=%s' % (period, alpha, eta)
        plt.plot(mock['x'], l1tf_fit['x'], marker='o', linestyle='-', markersize=4,alpha=0.8,label=lab)
        lab = 'L1TF + seasonal, period=%s, alpha=%s, eta=%s' % (period, alpha,eta)
        plt.plot(mock['x'], l1tf_fit['x_with_seasonal'], marker='o', markersize=4, linestyle='-', label=lab)
        plt.legend(loc='lower left')
        plt.ylim(0, 1)
        plt.show()


def test_l1tf_on_mock_with_period_l1p_with_spike(alpha=0.5, period=6, eta=0.1,
                    doplot=doplot, sea_amp=0.05):
    #no test for this yet
    return

    if doplot:
        plt.clf()
    mock = make_l1tf_mock(period=period, sea_amp=sea_amp)
    y = mock['y_with_seasonal']
    num = len(y)
    y[num/2] += 3.0
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
    assert_is_good(resid, mean_resid_max=1e-12, mean_abs_resid_max=0.6, max_abs_resid_max=1)

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




