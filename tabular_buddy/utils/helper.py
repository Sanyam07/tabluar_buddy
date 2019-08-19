# @Author: yican.kz
# @Date: 2019-08-19 16:23:49
# @Last Modified by:   yican.kz
# @Last Modified time: 2019-08-19 16:23:49


# Standard libraries
import os
import sys
import inspect
import hashlib
from pathlib import Path

# Third party libraries
import joblib
import numpy as np
import pandas as pd

# User defined libraries
from .file import mkdir


class ProgressBar:
    def __init__(self, n_batch, bar_len=80):
        """Brief description.

        Detailed description.

        Parameters
        ----------
        bar_len: int
            The length you want to display your bar.
        n_batch: int
            Total rounds to iterate.
        Returns
        -------
        None

        Examples
        --------
        import time
        progressBar = ProgressBar(100)

        for i in range(100):
            progressBar.step(i)
            time.sleep(0.1)
        """
        self.bar_len = bar_len
        self.progress_used = 0
        self.progress_remanent = bar_len
        self.n_batch = n_batch

    def step(self, i):
        self.progress_used = int(round(i * self.bar_len / self.n_batch))
        self.progress_remanent = self.bar_len - self.progress_used
        sys.stdout.write(
            "\r"
            + ">" * self.progress_used
            + "Epoch Progress: "
            + "{:.2%}".format((i) / self.n_batch)
            + "=" * self.progress_remanent
        )
        sys.stdout.flush()


def csv_2_pickle(paths):
    """ Convert csv files into pickle format files.

     Parameters
     ----------
     paths: list
        Csv file paths.

     Examples
     --------
     PATH = Path("data/raw/")
     CSV = [str(i) for i in list(PATH.glob("*.csv"))]
     csv_2_pickle(CSV)
    """
    PATH = Path("data/raw/")
    paths = [str(i) for i in list(PATH.glob("*.csv"))]

    for path in paths:
        data = pd.read_csv(path)
        data.columns = list(map(str.lower, data.columns))
        joblib.dump(data, path.split("csv")[0] + "p")


def generate_md5_token_from_dict(input_params):
    """ Generate distinct md5 token from a dictionary.
    初衷是为了将输入一个函数的输入参数内容编码为独一无二的md5编码, 方便在其变动的时候进行检测.

    Parameters
    ----------
    input_params : dict
        Dictionary to be encoded.

    Returns
    -------
    str
        Encoded md5 token from input_params.
    """
    input_params_token = ""
    # print(">>"*88)
    # print(input_params)
    # print(">>"*88)
    for v in list(input_params["kwargs"].values()) + list(input_params["args"]) + list(input_params["feature_path"]):
        if type(v) in [pd.DataFrame, pd.Series]:
            input_params_token += "pandas_" + str(v.memory_usage().sum()) + "_" + str(v.shape) + "_"
        elif type(v) in [np.ndarray]:
            input_params_token += "numpy_" + str(v.mean()) + "_" + str(v.shape) + "_"
        elif type(v) in [list, tuple, set]:
            input_params_token += "list_" + str(v) + "_"
        elif type(v) == str:
            input_params_token += "str_" + v + "_"
        elif type(v) in [int, float]:
            input_params_token += "numeric_" + str(v) + "_"
        elif type(v) == bool:
            input_params_token += "bool_" + str(v) + "_"
        elif type(v) == dict:
            input_params_token += "dict_" + str(v) + "_"
        else:
            raise "Add type {}".format(type(v))
    m = hashlib.md5(input_params_token.encode("gb2312")).hexdigest()
    return m


def cache(feature_path="data/features/"):
    # https://foofish.net/python-decorator.html
    def decorator(func):
        def wrapper(*args, **kwargs):
            # =========================================================================================================
            # 将输入所有输入参数的内容编码为md5, 以后每次检测是否变动, 缺陷(pandas类型)只检查了[内存]和[形状]).
            # =========================================================================================================
            input_params = locals().copy()
            feature_code_input_params_this_time = generate_md5_token_from_dict(input_params)

            # ==============================================================================================================
            # 检测保存数据和代码的文件夹是否存在, 如不存在则进行新建.
            # ==============================================================================================================
            function_name = func.__name__
            data_folder_path = feature_path + "data/"
            code_folder_path = feature_path + "code/"
            mkdir(data_folder_path)
            mkdir(code_folder_path)

            # 生成数据文件路径
            feature_data_path = data_folder_path + function_name + "_" + feature_code_input_params_this_time + ".p"
            is_data_cached = os.path.exists(feature_data_path)
            # 生成代码文件路径
            feature_code_path = code_folder_path + function_name + "_" + feature_code_input_params_this_time + ".p"
            is_code_cached = os.path.exists(feature_code_path)
            # 获取本次代码的内容
            feature_code_this_time = inspect.getsource(func)

            # 如果探测变动代码目录下没有 "函数代码文件" --> 重新运行一次这个函数, 并存储所有校正信息.
            if not is_code_cached:
                print("{} code file is not exist!".format(func.__name__))
                feature_data = func(*args, **kwargs)
                joblib.dump(feature_data, feature_data_path)
                joblib.dump(feature_code_this_time, feature_code_path)
                return feature_data

            # 如果 "函数代码文件" 变动 --> 重新运行一次这个函数, 并存储所有校正信息.
            feature_code_last_time = joblib.load(feature_code_path)
            flag_code_changed = feature_code_this_time != feature_code_last_time
            if flag_code_changed:
                print("{} code file has been changed!".format(func.__name__))
                feature_data = func(*args, **kwargs)
                joblib.dump(feature_data, feature_data_path)
                joblib.dump(feature_code_this_time, feature_code_path)
                return feature_data

            # 如果 "操作对象生成数据"不存在 --> 存储这次 "操作对象代码" 并生成 "操作对象生成数据"
            if not is_data_cached:
                print("{} feature file is not exist!".format(func.__name__))
                feature_data = func(*args, **kwargs)
                joblib.dump(feature_data, feature_data_path)
                joblib.dump(feature_code_this_time, feature_code_path)
                return feature_data

            feature_data = joblib.load(feature_data_path)
            print("Restore feature from {}".format(feature_data_path))
            return feature_data

        return wrapper

    return decorator
