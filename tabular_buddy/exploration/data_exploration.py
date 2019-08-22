import numpy as np
import pandas as pd
from scipy.stats import skew
from IPython.display import display


####################################################################################
# Display function
####################################################################################


def show_missing_info(df):
    """Show missing information

        Parameters
        ----------
        df: pandas dataframe
            Dataframe to be computed.
        Return
        ------
        df_info: pandas dataframe
            Dataframe contains missing information.
    """
    df_missing = df.isnull().sum().sort_values(ascending=False)
    df_info = pd.concat(
        [
            pd.Series(df_missing.index.tolist()),
            pd.Series(df_missing.values),
            pd.Series(df[df_missing.index].dtypes.apply(lambda x: str(x)).values),
            pd.Series((df_missing / df.shape[0]).values),
        ],
        axis=1,
        ignore_index=True,
    )
    df_info.columns = ["col_name", "missing_count", "col_type", "missing_rate"]

    return df_info


def show_skewnewss_info(df):
    """Show skewness information.

        Parameters
        ----------
        df: pandas dataframe
            Dataframe to be computed.
        Return
        ------
        df_info: pandas dataframe
            Dataframe contains missing information.
    """
    numeric_cols = df.columns[df.dtypes != "object"].tolist()
    skew_value = []

    for i in numeric_cols:
        skew_value += [skew(df[i])]
    df_info = pd.concat(
        [
            pd.Series(numeric_cols),
            pd.Series(df.dtypes[df.dtypes != "object"].apply(lambda x: str(x)).values),
            pd.Series(skew_value),
        ],
        axis=1,
    )
    df_info.columns = ["var_name", "col_type", "skew_value"]
    df_info.sort_values("skew_value", inplace=True, ascending=False)
    return df_info


####################################################################################
#                              UNIVERSAL BLOCK
####################################################################################


def ka_get_NC_col_names(data):
    """Get column names of category and numeric

        Parameters
        ----------
        data: dataframe

        Return:
        ----------
        numerics_cols: numeric column names
        category_cols: category column names

    """
    numerics_cols = data.select_dtypes(exclude=["O"]).columns.tolist()
    category_cols = data.select_dtypes(include=["O"]).columns.tolist()
    return numerics_cols, category_cols


def ka_remove_duplicate_cols(df, **kwargs):
    """Remove duplicate columns

       Parameters
       ----------
       df: pandas dataframe
          Features matrix

       **kwargs: all parameters in drop_duplicates function
           subset : column label or sequence of labels, optional
                Only consider certain columns for identifying duplicates, by
                default use all of the columns
           keep : {'first', 'last', False}, default 'first'
               - ``first`` : Drop duplicates except for the first occurrence.
               - ``last`` : Drop duplicates except for the last occurrence.
               - False : Drop all duplicates.
           take_last : deprecated
           inplace : boolean, default False
                Whether to drop duplicates in place or to return a copy
       Return
       ------
       new pandas dataframe with "unique columns" and "removed column names"

       Example
       -------
       data_1_unique, removed_cols = ka_remove_duplicate_cols(data_1[numeric_cols])
    """
    df_unique_columns = df.T.drop_duplicates(**kwargs).T
    return df_unique_columns, set(df.columns.tolist()) - set(df_unique_columns.columns.tolist())


####################################################################################
#                              CATEGORICAL BLOCK
####################################################################################


def k_cat_explore(x: pd.Series):
    unique_cnt = x.nunique()
    value_cnts = x.value_counts(dropna=False)

    print("num of unique counts: {}".format(unique_cnt))
    plt_value_cnts(value_cnts.iloc[:20], x.name)
    display(value_cnts.iloc[:20])

    return unique_cnt, value_cnts


def plt_value_cnts(value_cnts, name):
    ax = value_cnts.plot(kind="barh", figsize=(10, 7), color="coral", fontsize=13)
    ax.set_title(name)

    # create a list to collect the plt.patches data
    totals = []

    # find the values and append to list
    for i in ax.patches:
        totals.append(i.get_width())

    # set individual bar lables using above list
    total = sum(totals)

    # set individual bar lables using above list
    for i in ax.patches:
        # get_width pulls left or right; get_y pushes up or down
        ax.text(
            i.get_width() * 1,
            i.get_y() + 0.3,
            str(round((i.get_width() / total) * 100, 2)) + "%",
            fontsize=15,
            color="black",
        )

    # invert for largest on top
    ax.invert_yaxis()
    ax.plot()


def ka_C_Binary_ratio(y, positive=1):
    """Find the positive ration of dependent variable

        Parameters
        ----------
        y: pandas series
           binary dependent variable
        positive: 1 or 0
                  identify which value is positive

        Return
        ------
        float value display positive rate
    """
    return y.value_counts()[positive] / (y.value_counts().sum())
