# @Author: yican.kz
# @Date: 2019-08-18 23:47:49
# @Last Modified by:   yican.kz
# @Last Modified time: 2019-08-18 23:47:49

# Standard libraries
import os
import time
from datetime import datetime


class Timer:
    """Reference : https://github.com/davidcpage/cifar10-fast/blob/master/core.py.

    Returns
    -------
    [type]
        [description]
    """

    def __init__(self, synch=None):
        self.synch = synch or (lambda: None)
        self.synch()
        self.times = [time.time()]
        self.total_time = 0.0

    def __call__(self, include_in_total=True):
        self.synch()
        self.times.append(time.time())
        delta_t = self.times[-1] - self.times[-2]
        if include_in_total:
            self.total_time += delta_t
        return delta_t


def time_func(func):
    """ Decorator for measuring time elapsed of function."""

    def wrapper(*args, **kwargs):
        start_time = time.time()
        func(*args, **kwargs)
        print("Executed in {} seconds".format(int(time.time() - start_time)))
        return

    return wrapper


def now(for_logging=True):
    """ Get Current time

    Parameters
    ----------
    for_logging : bool, optional
        Used for logging or used for folder name, by default True.

    Returns
    -------
    current_time : str
        Current time in string format.
    """
    if for_logging is True:
        current_time = strftime("%Y-%m-%d %H:%M:%S", localtime())
    else:
        current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    return current_time


def log_start_time(fname):
    global st_time
    st_time = time.time()
    print(
        """
# =================================================================================
# START !!! {}    PID: {}    Time: {}
# =================================================================================
""".format(
            fname.split("/")[-1], os.getpid(), datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
    )

    #    send_line(f'START {fname}  time: {elapsed_minute():.2f}min')

    return


def log_end_time(fname):
    print(
        """
# =================================================================================
# SUCCESS !!! {}  Total Time: {} seconds
# =================================================================================
""".format(
            fname.split("/")[-1], int(time.time() - st_time)
        )
    )
    return
