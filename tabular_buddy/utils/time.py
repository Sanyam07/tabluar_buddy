# @Author: yican.kz
# @Date: 2019-08-18 23:47:49
# @Last Modified by:   yican.kz
# @Last Modified time: 2019-08-18 23:47:49

# Standard libraries
import os
import time
from datetime import datetime
from time import strftime, localtime


def timeit(func):
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
        current_time = strftime("%Y-%m-%d_%H-%M-%S", localtime())
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
