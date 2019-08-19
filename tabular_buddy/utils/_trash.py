# @Author: yican.kz
# @Date: 2019-08-18 23:45:03
# @Last Modified by:   yican.kz
# @Last Modified time: 2019-08-18 23:45:03

# Standard libraries
import json
import argparse
from itertools import chain

# Third party libraries
import numpy as np
import pandas as pd
from IPython.display import display

INT8_MIN = np.iinfo(np.int8).min
INT8_MAX = np.iinfo(np.int8).max
INT16_MIN = np.iinfo(np.int16).min
INT16_MAX = np.iinfo(np.int16).max
INT32_MIN = np.iinfo(np.int32).min
INT32_MAX = np.iinfo(np.int32).max

FLOAT16_MIN = np.finfo(np.float16).min
FLOAT16_MAX = np.finfo(np.float16).max
FLOAT32_MIN = np.finfo(np.float32).min
FLOAT32_MAX = np.finfo(np.float32).max


def memoryUsage(data, detail=1):
    """Got memory usage of dataset
        Parameters
        ----------
        data: dataFrame

    """
    if detail:
        display(data.memory_usage())
    memory = data.memory_usage().sum() / (1024 * 1024)
    print("Memory usage : {0:.2f}MB".format(memory))
    return memory


def compressDataset(data):
    """
        Compress dataset using strategy below
        有个缺点，如果压缩一个数据到int8，那么对于所有大于int8_max的赋值，都会出问题
        FLOAT64
            一级一级往下找 FLOAT16 FLOAT32
        INT64
            # 如果最小值大于等于0，从unsigned格式里面找 这步暂时不做
            如果最小值小于0，从signed格式里面找
        Parameters
        ----------
        path: pandas Dataframe

        Returns
        -------
            None
    """
    memory_before_compress = memoryUsage(data, 0)
    print()
    length_interval = 50
    length_float_decimal = 4
    length_interval_half = np.int(length_interval / 2)
    num_cols = len(data.columns)

    print("=" * length_interval)

    for progress, col in enumerate(data.columns):
        col_dtype = data[col][:100].dtype

        if col_dtype != "object":
            print("Name: {0:24s} Type: {1}".format(col, col_dtype))
            col_series = data[col]
            col_min = col_series.min()
            col_max = col_series.max()

            if col_dtype == "float64":
                print(
                    " variable min: {0:15s} max: {1:15s}".format(
                        str(np.round(col_min, length_float_decimal)), str(np.round(col_max, length_float_decimal))
                    )
                )
                if (col_min > FLOAT16_MIN) and (col_max < FLOAT16_MAX):
                    data[col] = data[col].astype(np.float16)
                    print("  float16 min: {0:15s} max: {1:15s}".format(str(FLOAT16_MIN), str(FLOAT16_MAX)))
                    print("compress float64 --> float16")
                elif (col_min > FLOAT32_MIN) and (col_max < FLOAT32_MAX):
                    data[col] = data[col].astype(np.float32)
                    print("  float32 min: {0:15s} max: {1:15s}".format(str(FLOAT32_MIN), str(FLOAT32_MAX)))
                    print("compress float64 --> float32")
                else:
                    pass
                memory_after_compress = memoryUsage(data, 0)
                print(
                    "Compress Rate: [{0:.2%}]".format(
                        (memory_before_compress - memory_after_compress) / memory_before_compress
                    )
                )
                print("=" * length_interval_half + "{:.2%}".format(progress / num_cols) + "=" * length_interval_half)

            if col_dtype == "int64":
                print(" variable min: {0:15s} max: {1:15s}".format(str(col_min), str(col_max)))
                type_flag = 64
                if (col_min > INT8_MIN / 2) and (col_max < INT8_MAX / 2):
                    type_flag = 8
                    data[col] = data[col].astype(np.int8)
                    print("     int8 min: {0:15s} max: {1:15s}".format(str(INT8_MIN), str(INT8_MAX)))
                elif (col_min > INT16_MIN) and (col_max < INT16_MAX):
                    type_flag = 16
                    data[col] = data[col].astype(np.int16)
                    print("    int16 min: {0:15s} max: {1:15s}".format(str(INT16_MIN), str(INT16_MAX)))
                elif (col_min > INT32_MIN) and (col_max < INT32_MAX):
                    type_flag = 32
                    data[col] = data[col].astype(np.int32)
                    print("    int32 min: {0:15s} max: {1:15s}".format(str(INT32_MIN), str(INT32_MAX)))
                    type_flag = 1
                else:
                    pass
                memory_after_compress = memoryUsage(data, 0)
                print(
                    "Compress Rate: [{0:.2%}]".format(
                        (memory_before_compress - memory_after_compress) / memory_before_compress
                    )
                )
                if type_flag == 32:
                    print("compress (int64) ==> (int32)")
                elif type_flag == 16:
                    print("compress (int64) ==> (int16)")
                else:
                    print("compress (int64) ==> (int8)")
                print("=" * length_interval_half + "{:.2%}".format(progress / num_cols) + "=" * length_interval_half)

    print()
    memory_after_compress = memoryUsage(data, 0)
    print("Compress Rate: [{0:.2%}]".format((memory_before_compress - memory_after_compress) / memory_before_compress))


# ==============================================================================
# 展示相关
# ==============================================================================
def display_pro(data: pd.DataFrame, n=5):
    """Pro version of display function.

     Display [memory usage], [data shape] and [first n rows] of a pandas dataframe.

     Parameters
     ----------
     data: pandas dataframe
        Pandas dataframe to be displayed.
     n: int
        First n rows to be displayed.

     Example
     -------
     import pandas as pd
     from sklearn.datasets import load_boston
     data = load_boston()
     data = pd.DataFrame(data.data)
     display_pro(data)

        Parameters
        ----------
        data: pandas dataframe


        Returns
        -------
            None
    """
    _ = memory_usage(data, 0)
    print("Data shape   : {}".format(data.shape))
    display(data[:n])


def memory_usage(data: pd.DataFrame, detail=1):
    """Show memory usage.

     Parameters
     ----------
     data: pandas dataframe
     detail: int, optinal (default = 1)
        0: show memory of each column
        1: show total memory

     Examples
     --------
     import pandas as pd
     from sklearn.datasets import load_boston
     data = load_boston()
     data = pd.DataFrame(data.data)
     memory = memory_usage(data)
     """

    memory_info = data.memory_usage()
    if detail:
        display(memory_info)

    if type(memory_info) == int:
        memory = memory_info / (1024 * 1024)
    else:
        memory = data.memory_usage().sum() / (1024 * 1024)
    print("Memory usage : {0:.2f}MB".format(memory))
    return memory


# ==============================================================================
# 功能相关
# ==============================================================================
def find_a_not_in_b(a: list, b: list):
    """找到a中存在b中不存在的元素

        Parameter
        ---------

        Return
        ------
        list
    """
    # isinstance(a, float)是为了防止[空值]的情况
    if isinstance(a, float) or isinstance(b, float):
        return a
    return [i1 for i1 in a if i1 not in b]


def find_a_in_b(a: list, b: list):
    """找到a中存在b中也存在的元素

        Parameter
        ---------

        Return
        ------
        list
    """
    # isinstance(a, float)是为了防止[空值]的情况
    if isinstance(a, float) or isinstance(b, float):
        return []
    return [i1 for i1 in a if i1 in b]


def sum_list_in_list(list_in_list):
    """将 list in list 变为 list
        Parameter
        ---------
        list_in_list: list

        Return
        ------
        list

        Example
        -------
        sum_list([['1','2'], ['2']]) -> ['1', '2', '2']
    """
    return list(chain.from_iterable(list_in_list))


def unique_list_in_list(list_in_list):
    """将 list in list 变为 list
        Parameter
        ---------
        list_in_list: list

        Return
        ------
        list

        Example
        -------
        unique_list_in_list([['1','2'], ['2']]) -> ['1', '2']
    """
    li = sum_list_in_list(list_in_list)
    return list(set(li))


def is_primary_key(data, column_list):
    """Verify if columns in column list can be treat as primary key

        Parameter
        ---------
        data: pandas dataframe

        column_list: list_like
                     column names in a list

        Return
        ------
        boolean: if true, these columns are unique in combination and can be used as a key
                 if false, these columns are not unique in combination and can not be used as a key
    """

    return data.shape[0] == data.groupby(column_list).size().reset_index().shape[0]


def is_identical(col_1, col_2):
    """判断数据集的两列是否一致

        Parameters
        ----------
        data: dataframe
        col_1: string
        col_2: string

        Return
        ------
        True、False
    """
    if sum(col_1 == col_2) / len(col_1) == 1:
        return True
    return False


def reverse_dict(x: dict):
    """翻转一个dict

        Parameters
        ----------
        x: dict

        Return
        ------
        dict
    """
    return {v: k for k, v in x.items()}


def min_max(x):
    return min(x), max(x)


# ==============================================================================================================
# Others
# ==============================================================================================================
def load_config(path):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=path)
    options = parser.parse_args()
    config = json.load(open(options.config))
    return config


# def ka_erfinv_rank_transform(x):
#     '''
#         This is used on numeric variable, after doing this operation, one should do MM and SS on all dataset.
#     '''
#     mm = MinMaxScaler()
#     tmp = erfinv(np.clip(np.squeeze(mm.fit_transform(rankdata(x).reshape(-1,1))), 0, 0.999999999))
#     tmp = tmp - np.mean(tmp)
#     return tmp
#
# def kaggle_points(n_teams, n_teammates, rank, t=1):
#     return (100000 / np.sqrt(n_teammates)) * (rank ** (-0.75)) * (np.log10(1 + np.log10(n_teams))) * (np.e**(t/500))
