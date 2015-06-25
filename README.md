myl1tf
======

L1TF trend fitting in python including seasonality and other sparse features.
Includes a novel "l1-everything" algorithm which uses l1-norms for everything
including the residual.

Example:

test_myl1tf.test_l1tf_on_mock_with_period(period=6,alpha=0.5,eta=1.0)

![](https://github.com/dave31415/myl1tf/blob/master/example.png)

Based on the [paper](
http://web.stanford.edu/%7Egorin/papers/l1_trend_filter.pdf
) By Kim, Koh & Boyd

Requires numpy and cvxopt libraries
matplotlib required to enable plotting
