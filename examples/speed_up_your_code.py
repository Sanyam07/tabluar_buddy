# @Author: yican.kz
# @Date: 2019-08-23 23:38:16
# @Last Modified by:   yican.kz
# @Last Modified time: 2019-08-23 23:38:16

# Standard libraries
import os
import sys
import random

# Third party libraries
import pandas as pd
import numpy as np
from numba import jit

sys.path.insert(0, os.path.abspath(".."))
from tabular_buddy.utils import parallelize, Timer


def monte_carlo_pi(nsamples):
    acc = 0
    for i in range(nsamples):
        x = random.random()
        y = random.random()
        if (x ** 2 + y ** 2) < 1.0:
            acc += 1
    return 4.0 * acc / nsamples


@jit(nopython=True)
def monte_carlo_pi_numba(nsamples):
    acc = 0
    for i in range(nsamples):
        x = random.random()
        y = random.random()
        if (x ** 2 + y ** 2) < 1.0:
            acc += 1
    return 4.0 * acc / nsamples


def func(data):
    return data["a"].apply(lambda x: np.sqrt(x)) - data["b"] ** 2


if __name__ == "__main__":
    # ==============================================================================================================
    # Using numba to speed up your function, [21 times!]
    # -----------------------------------------
    # Before speed up : Res is 3.1414 | 21 seconds
    # After speed up  : Res is 3.1419 | 1 seconds
    # ==============================================================================================================
    nsamples = 10000000
    data = pd.DataFrame({"a": np.random.rand(nsamples), "b": np.random.rand(nsamples)})

    tick_tock = Timer()
    print("Before speed up : Res is {:.4f} | {} seconds".format(monte_carlo_pi(50000000), int(tick_tock())))
    print("After speed up  : Res is {:.4f} | {} seconds".format(monte_carlo_pi_numba(50000000), int(tick_tock())))

    # ==============================================================================================================
    # Using python multiprocessing to speed up your function, [3 times!]
    # Actual performance depends on your machine
    # -----------------------------------------
    # Before speed up : Res 3333598.7172 | 16 seconds
    # After speed up  : Res 3333598.7172 | 5 seconds
    # ==============================================================================================================
    res = func(data)
    print("Before speed up : Res {:.4f} | {} seconds".format(res.sum(), int(tick_tock())))
    res = parallelize(data, func)
    print("After speed up  : Res {:.4f} | {} seconds".format(res.sum(), int(tick_tock())))
