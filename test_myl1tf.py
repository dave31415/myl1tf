import myl1tf
import numpy as np
from matplotlib import pylab as plt

def make_l1tf_mock(doplot=True, period=6, sea_amp=0.05):
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
        plt.plot(x, y, marker='o', linestyle='-', label='True', markersize=10, alpha=0.3)
        plt.plot(x, y_with_seasonal, marker='o', linestyle='-', label='True w_ seasonal')

    return {'x': x, 'y': y, 'y_with_seasonal': y_with_seasonal, 'seas_lookup': seas_lookup}


def test_l1tf_on_mock(alpha=1.0):
    mock = make_l1tf_mock()
    l1tf_fit = myl1tf.l1tf(mock['y'], alpha=alpha)
    plt.plot(mock['x'], l1tf_fit, marker='o', linestyle='-', label='L1TF Dual, alpha=%s' % alpha)
    plt.legend(loc='lower center')