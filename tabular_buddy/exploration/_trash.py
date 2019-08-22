# from IPython.display import display_html
# from pandas_summary import DataFrameSummary


# def _ka_display_col_type(data):
#     """See column type distribution

#        Parameters
#        ----------
#        data: pandas dataframe

#        Return
#        ------
#        dataframe
#     """
#     column_type = data.dtypes.reset_index()
#     column_type.columns = ["count", "column type"]
#     return column_type.groupby(["column type"]).agg("count").reset_index()


# def ka_display_side_by_side(*args):
#     html_str = ""
#     for df in args:
#         html_str += df.to_html()
#     display_html(html_str.replace("table", 'table style="display:inline"'), raw=True)


# def ka_display_muti_tables_summary(tables, table_names, n=5):
#     """display multi tables' summary

#         Parameters
#         ----------
#         tables: list_like
#                 Pandas dataframes
#         table_names: list_like
#                      names of each dataframe

#         Return
#         ------
#         1. show head of data
#         2. show column types of data
#         3. show summary of data
#     """
#     for t, t_name in zip(tables, table_names):
#         print(t_name + ":", t.shape)
#         ka_display_side_by_side(t.head(n=n), _ka_display_col_type(t), DataFrameSummary(t).summary())


# def ka_display_groupby_n_1_stats(data, group_columns_list, target_columns_list):
#     """Evaluate statistical indicators in each category

#        Parameters
#        ----------
#        data: pandas dataframe
#           Features matrix
#        group_columns_list: list_like
#           List of columns you want to group with, could be multiple columns
#        target_columns_list: list_like
#           column you want to compute stats, need to be a list with only one element

#        Return
#        ------
#        pandas dataframe

#        Example
#        -------
#        df = ka_display_groupby_n_1_stats(train, ['class'], ['translate_flag'])
#     """

#     grouped = data.groupby(group_columns_list)
#     df = grouped[target_columns_list].agg([len, np.mean, np.median, np.min, np.max, np.std]).reset_index()
#     df.columns = df.columns.droplevel(0)
#     df["percent"] = df.len * 100 / df.len.sum()
#     df["percent"] = pd.Series(["{0:.2f}%".format(val) for val in df["percent"]], index=df.index)

#     return df.sort_values("mean", ascending=False)
