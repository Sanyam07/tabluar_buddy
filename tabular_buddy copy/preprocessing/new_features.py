import warnings
import numpy as np
import pandas as pd

from gensim.models import Word2Vec

from ..utils.helper import tick_tock
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import StratifiedKFold
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


def _deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used."""

    def new_func(*args, **kwargs):
        warnings.warn("Call to deprecated function {}.".format(func.__name__), category=DeprecationWarning)
        return func(*args, **kwargs)

    new_func.__name__ = func.__name__
    new_func.__doc__ = func.__doc__
    new_func.__dict__.update(func.__dict__)
    return new_func


def create_groupby_features(
    df,
    group_columns_list,
    method_dict,
    add_to_original_data=False,
    verbose=1,
    verbose_detail="create stats features",
    suffix="",
):
    """Create statistical columns, group by [N columns] and compute stats on [N column]

       Parameters
       ----------
       df: pandas dataframe
          Features matrix
       group_columns_list: list_like
          List of columns you want to group with, could be multiple columns
       method_dict: python dictionary
          Dictionay used to create stats variables
          shoubld be {'feature': ['method_1', 'method_2']}, if method is a lambda, use function inplace.
       add_to_original_data: boolean
          only keep stats or add stats variable to raw data
       verbose: int
          1 return tick_tock info 0 do not return any info
       Return
       ------
       new pandas dataframe with original columns and new added columns

       Example
       -------
       ka_add_groupby_features(data
                               ,['class']
                               ,{'before': ['count','mean']})

       Update
       ------
       2017/09/26: pandas 0.20.3 has deprecate using just a dict to rename and create stat variables
       ,so I add another parameter method_list to fix this warning.

       2017/9/27: add verbose_detail parameter, let user specify infos they want print.
       2017/10/8: generate column names automatic
    """
    with tick_tock(verbose_detail, verbose):
        try:
            if type(group_columns_list) == list:
                pass
            else:
                raise TypeError(group_columns_list, "should be a list")
        except TypeError as e:
            print(e)
            raise

        df_new = df.copy()
        grouped = df_new.groupby(group_columns_list)

        the_stats = grouped.agg(method_dict)
        if suffix != "":
            the_stats.columns = [
                "".join(group_columns_list) + "_LV_" + "_".join(x[::-1]) + "_" + str(suffix)
                for x in the_stats.columns.ravel()
            ]
        else:
            the_stats.columns = [
                "".join(group_columns_list) + "_LV_" + "_".join(x[::-1]) for x in the_stats.columns.ravel()
            ]
        the_stats.reset_index(inplace=True)

        if not add_to_original_data:
            df_new = the_stats
        else:
            df_new = pd.merge(
                left=df_new[group_columns_list], right=the_stats, on=group_columns_list, how="left"
            ).reset_index(drop=True)

    return df_new


@_deprecated
def ka_create_groupby_features_old(df, group_columns_list, agg_dict, keep_only_stats=True, verbose=1):
    """Create statistical columns, group by [N columns] and compute stats on [N column]
       Parameters
       ----------
       df: pandas dataframe
          Features matrix
       group_columns_list: list_like
          List of columns you want to group with, could be multiple columns
       agg_dict: python dictionary
          Dictionay used to create stats variables
       keep_only_stats: boolean
          only keep stats or return both raw columns and stats
       verbose: int
          1 return tick_tock info 0 do not return any info
       Return
       ------
       new pandas dataframe with original columns and new added columns
       Example
       -------
       {real_column_name: {your_specified_new_column_name : method}}
       agg_dict = {'user_id':{'prod_tot_cnts':'count'},
                   'reordered':{'reorder_tot_cnts_of_this_prod':'sum'},
                   'user_buy_product_times': {'prod_order_once':lambda x: sum(x==1),
                                              'prod_order_more_than_once':lambda x: sum(x==2)}}
       ka_add_stats_features_1_vs_n(train, ['product_id'], agg_dict)
    """
    with tick_tock("add stats features", verbose):
        try:
            if type(group_columns_list) == list:
                pass
            else:
                raise TypeError(k + "should be a list")
        except TypeError as e:
            print(e)
            raise

        df_new = df.copy()
        grouped = df_new.groupby(group_columns_list)

        the_stats = grouped.agg(agg_dict)
        the_stats.columns = the_stats.columns.droplevel(0)
        the_stats.reset_index(inplace=True)
        if keep_only_stats:
            df_new = the_stats
        else:
            df_new = pd.merge(left=df_new, right=the_stats, on=group_columns_list, how="left")

    return df_new


def ka_replace_hash(hashes, hash_id_table):
    """Replace "hash in hashes" to "numeric index in hash_id_table"

       Parameter
       ---------
       hashes: pandas series
       hash_id_table: pandas series

       Return
       ------
       numpy array:
           replaced numeric number


       Example
       -------
       user_ids:
       0        d9dca3cb44bab12ba313eaa681f663eb
       1        560574a339f1b25e57b0221e486907ed

       detail.USER_ID_hash:
       0         d9dca3cb44bab12ba313eaa681f663eb
       1         560574a339f1b25e57b0221e486907ed

       replace_hash(detail.USER_ID_hash, user_ids)
    """
    replace_table = pd.Series(hash_id_table.index, index=hash_id_table.values)
    return replace_table[hashes].values


def ka_add_hash_feature(df, category_columns_list):
    """Create hash column unique in your specified columns

       Parameters
       ----------
       df: pandas dataframe
           Features matrix

       category_columns_list: list_like
           column names in a list

       Return
       ------
       new pandas dataframe with original columns and new added columns
    """
    with tick_tock("add hash feature"):
        df_new = df.copy()
        if len(category_columns_list) > 8:
            df_new["hash_" + category_columns_list[0] + "_" + category_columns_list[-1]] = df_new[
                category_columns_list
            ].apply(lambda x: hash(tuple(x)), axis=1)
        else:
            df_new["hash_" + "".join(category_columns_list)] = df_new[category_columns_list].apply(
                lambda x: hash(tuple(x)), axis=1
            )
    return df_new


def create_svd_interaction_features(
    data, col_tobe_grouped, col_tobe_computed, tfidf=True, n_components=1, verbose=False
):
    """Extract col_tobe_grouped level information utilize information of col_tobe_computed by using SVD.

    Parameters
    ----------
    data : pandas dataframe
    col_tobe_grouped : list
        [str, str, str, ...]
    col_tobe_computed : str
    tfidf : bool
        If true, use tfidf to extract information
        If false, use count to extract information
    n_components: int
        Number of columns to genderate
    verbose: bool
        If true, show debug information.
        If false, do not show debug information.

    Returns
    -------
    result : pandas dataframe
        col_tobe_grouped level dataframe, columns are information about col_tobe_computed.

    Examples
    --------
    Your code here.
    """

    if verbose:
        print("col_tobe_grouped:{} | col_tobe_computed:{}".format(col_tobe_grouped, col_tobe_computed))
        print("dataset shape: {}".format(data.shape))

    # Step1: Generate dataframe that to be embedded
    data_tobe_embedded = data.groupby(col_tobe_grouped)[col_tobe_computed].agg(
        lambda x: " ".join(list([str(y) for y in x]))
    )
    if verbose:
        print("\nData shape to be embedded: {}".format(data_tobe_embedded.shape))
        print(data_tobe_embedded[:2])

    # Step2: Choose appropriate vectorizer
    if tfidf:
        vectorizer = TfidfVectorizer(tokenizer=lambda x: x.split(" "))
    else:
        vectorizer = CountVectorizer(tokenizer=lambda x: x.split(" "))

    # Step3: Create vectorizer
    data_embedded_vector = vectorizer.fit_transform(data_tobe_embedded)
    if verbose:
        print("\nData shape embedded vector: {}".format(data_embedded_vector.shape))

    # Step4: Embed information of col_tobe_computed into col_tobe_grouped level
    svd = TruncatedSVD(n_components=n_components, random_state=2019)
    data_embedded_reduce = svd.fit_transform(data_embedded_vector)
    result = pd.DataFrame(data_embedded_reduce)
    if tfidf:
        result.columns = [
            "_".join(col_tobe_grouped) + "_{}_svd_tfidf_{}".format(col_tobe_computed, index)
            for index in range(n_components)
        ]
    else:
        result.columns = [
            "_".join(col_tobe_grouped) + "_{}_svd_count_{}".format(col_tobe_computed, index)
            for index in range(n_components)
        ]
    result[col_tobe_grouped] = data_tobe_embedded.reset_index()[col_tobe_grouped]
    if verbose:
        print("Data shape embedded svd: {}".format(data_embedded_reduce.shape))
        print(result[:2])

    return result


def create_w2v_interaction_features(data, col1, col2, n_components, window_size, verbose=False):
    """Extract col1 level information utilize information of col2 by using word2vec.

    Parameters
    ----------
    data : pandas dataframe
    col1 : str
    col2 : str
    n_components: int
        Number of columns to genderate.
    window_size: int
        Window size of word2vec method.
    verbose: bool
        If true, show debug information.
        If false, do not show debug information.

    Returns
    -------
    result : pandas dataframe
        col1 level dataframe, columns are information about col2.

    Examples
    --------
    Your code here.
    """

    if verbose:
        print("col1:{} | col2:{}".format(col1, col2))
        print("dataset shape: {}".format(data.shape))

    # Step1: Generate dataframe that to be embedded.
    data_tobe_embedded = data.groupby([col2])[col1].agg(lambda x: list([str(y) for y in x]))
    list_tobe_embedded = list(data_tobe_embedded.values)
    if verbose:
        print("\nData shape to be embedded: {}".format(data_tobe_embedded.shape))
        print(data_tobe_embedded[:2])

    # Step2: Do word embedding.
    w2v = Word2Vec(list_tobe_embedded, size=n_components, window=window_size, min_count=1)
    keys = list(w2v.wv.vocab.keys())
    dict_w2v = {}
    for key in keys:
        dict_w2v[key] = w2v.wv[key]
    result = pd.DataFrame(dict_w2v).T.reset_index()

    # Step3: Rename new columns/
    result.columns = [col1] + [col1 + "_{}_w2v_{}".format(col2, index) for index in range(n_components)]
    result[col1] = result[col1].astype(data[col1].dtype)
    return result


class TargetEncodingSmoothing(BaseEstimator, TransformerMixin):
    def __init__(self, columns_names, k, f):
        """ Target encoding class.
        
        Parameters
        ----------
        columns_names : list
            Columns to be encoded.
        k : float
            Inflection point, that's the point where  f(x)  is equal 0.5.
        f : float
            Steepness, a value which controls how step is our function.
        """
        self.columns_names = columns_names
        self.learned_values = {}
        self.dataset_mean = np.nan
        self.k = k
        self.f = f

    def smoothing_func(self, N):
        return 1 / (1 + np.exp(-(N - self.k) / self.f))

    def fit(self, X, y, **fit_params):
        """ Fit target encodings.
        
        Parameters
        ----------
        X : pandas.DataFrame
            Pandas dataframe which contains features.
        y : numpy
            Target values.
        
        Returns
        -------
        Class
            
        """
        X_ = X.copy()
        X_["__target__"] = y
        self.learned_values = {}
        self.dataset_mean = np.mean(y)

        for c in [x for x in X_.columns if x in self.columns_names]:
            stats = X_[[c, "__target__"]].groupby(c)["__target__"].agg(["mean", "size"])
            # Compute weight.
            stats["alpha"] = self.smoothing_func(stats["size"])
            # Take weighted sum of 2 means: dataset mean and level mean.
            stats["__target__"] = stats["alpha"] * stats["mean"] + (1 - stats["alpha"]) * self.dataset_mean
            # Keep weighted target and raw encoded columns.
            stats = stats.drop([x for x in stats.columns if x not in ["__target__", c]], axis=1).reset_index()
            # Save into dict
            self.learned_values[c] = stats
        return self

    def transform(self, X, **fit_params):
        """ Transform fitted target encoding information into X.
        
        Parameters
        ----------
        X : pandas.DataFrame
            Pandas dataframe which contains features.
        
        Returns
        -------
        pandas.DataFrame
            Transformed values.
        """
        # Get raw values.
        transformed_X = X[self.columns_names].copy()
        # Transform encoded information into raw values.
        for c in transformed_X.columns:
            transformed_X[c] = transformed_X[[c]].merge(self.learned_values[c], on=c, how="left")["__target__"]
        # Fill y dataset mean into missing values.
        transformed_X = transformed_X.fillna(self.dataset_mean)
        transformed_X.columns = [d + "_smooth_te" for d in transformed_X.columns]
        return transformed_X

    def fit_transform(self, X, y, **fit_params):
        """ Fit and Transform
        
        Parameters
        ----------
        X : pandas.DataFrame
            Pandas dataframe which contains features.
        y : numpy array
            Target values.
        
        Returns
        -------
        pandas.DataFrame
            Transformed values.
        """

        self.fit(X, y)
        return self.transform(X)


def get_CV_target_encoding(data, y, encoder, cv=5):
    """ Add cross validation noise into training target encoding.
    Parameters
    ----------
    data : pandas.DataFrame
        Pandas dataframe which contains features.
    y : numpy array
        Target values.
    encoder : TargetEncodingSmoothing
        TargetEncodingSmoothing Instance
    cv : int, optional
        Cross validation fold, by default 5
    
    Returns
    -------
    [type]
        [description]
    """
    # Create cross validation schema.
    skf = StratifiedKFold(n_splits=cv, random_state=2019, shuffle=True)
    result = []

    # Do cross validation.
    for train_index, test_index in skf.split(data, y):
        encoder.fit(data.iloc[train_index, :].reset_index(drop=True), y[train_index])
        tmp = encoder.transform(data.iloc[test_index, :].reset_index(drop=True))
        tmp["index"] = test_index
        result.append(tmp)

    # Concat all folds.
    result = pd.concat(result, ignore_index=True)
    # Recover to default order.
    result = result.sort_values("index").reset_index(drop=True).drop("index", axis=1)

    return result


class TargetEncodingExpandingMean(BaseEstimator, TransformerMixin):
    def __init__(self, columns_names):
        self.columns_names = columns_names
        self.learned_values = {}
        self.dataset_mean = np.nan

    def fit(self, X, y, **fit_params):
        X_ = X.copy()
        self.learned_values = {}
        self.dataset_mean = np.mean(y)
        X_["__target__"] = y
        for c in [x for x in X_.columns if x in self.columns_names]:
            stats = X_[[c, "__target__"]].groupby(c)["__target__"].agg(["mean", "size"])
            stats["__target__"] = stats["mean"]
            stats = stats.drop([x for x in stats.columns if x not in ["__target__", c]], axis=1).reset_index()
            self.learned_values[c] = stats
        return self

    def transform(self, X, **fit_params):
        transformed_X = X[self.columns_names].copy()
        for c in transformed_X.columns:
            transformed_X[c] = (transformed_X[[c]].merge(self.learned_values[c], on=c, how="left"))["__target__"]
        transformed_X = transformed_X.fillna(self.dataset_mean)
        transformed_X.columns = [d + "_expand_te" for d in transformed_X.columns]
        return transformed_X

    def fit_transform(self, X, y, **fit_params):
        self.fit(X, y)

        # Expanding mean transform
        X_ = X[self.columns_names].copy().reset_index(drop=True)
        X_["__target__"] = y
        X_["index"] = X_.index
        X_transformed = pd.DataFrame()
        for c in self.columns_names:
            X_shuffled = X_[[c, "__target__", "index"]].copy()
            X_shuffled = X_shuffled.sample(n=len(X_shuffled), replace=False)
            X_shuffled["cnt"] = 1
            X_shuffled["cumsum"] = X_shuffled.groupby(c, sort=False)["__target__"].apply(lambda x: x.shift().cumsum())
            X_shuffled["cumcnt"] = X_shuffled.groupby(c, sort=False)["cnt"].apply(lambda x: x.shift().cumsum())
            X_shuffled["encoded"] = X_shuffled["cumsum"] / X_shuffled["cumcnt"]
            X_shuffled["encoded"] = X_shuffled["encoded"].fillna(self.dataset_mean)
            X_transformed[c] = X_shuffled.sort_values("index")["encoded"].values

        X_transformed.columns = [d + "_expand_te" for d in X_transformed.columns]
        return X_transformed


def create_expand_noise_te_features(df_train, y_train, df_test, columns_names):
    """[summary]
    
    Parameters
    ----------
    df_train : pandas.DataFrame
        Pandas dataframe which contains features.
    y_train : numpy array
        Train target
    df_test : pandas.DataFrame
        Pandas dataframe which contains features.
    columns_names : list
        Columns to be encoded.
    k : float
        Inflection point, that's the point where  f(x)  is equal 0.5.
    f : float
        Steepness, a value which controls how step is our function.
    cv_noise : int, optional
        [description], by default 5
    
    Returns
    -------
    [type]
        [description]
    """
    te = TargetEncodingExpandingMean(columns_names=columns_names)
    X_train = te.fit_transform(df_train, y_train)

    X_test = te.transform(df_test)

    return X_train, X_test


def create_smooth_noise_te_features(df_train, y_train, df_test, columns_names, k, f, cv_noise=5):
    """[summary]
    
    Parameters
    ----------
    df_train : pandas.DataFrame
        Pandas dataframe which contains features.
    y_train : numpy array
        Train target
    df_test : pandas.DataFrame
        Pandas dataframe which contains features.
    columns_names : list
        Columns to be encoded.
    k : float
        Inflection point, that's the point where  f(x)  is equal 0.5.
    f : float
        Steepness, a value which controls how step is our function.
    cv_noise : int, optional
        [description], by default 5
    
    Returns
    -------
    [type]
        [description]
    """
    te = TargetEncodingSmoothing(columns_names=columns_names, k=k, f=f)
    X_train = get_CV_target_encoding(df_train, y_train, te, cv=cv_noise)

    te.fit(df_train, y_train)
    X_test = te.transform(df_test)

    return X_train, X_test


def create_noise_te_features_forlocal_cv(data, y, columns_names, k, f, n_splits=5, cv_noise=5):
    """ Load features and target, then generate target encoded values to correspoding train and valid.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Pandas dataframe which contains features.
    y : numpy array
        Target values.
    columns_names : list
        Columns to be encoded.
    k : float
        Inflection point, that's the point where  f(x)  is equal 0.5.
    f : float
        Steepness, a value which controls how step is our function.
    n_splits : int optional
        Cross validation fold, by default 5
    cv_noise : int optional
        Noise cross validation fold, by default 5
    
    Returns
    -------
    X_train : pandas.DataFrame
        Train encoded columns.
    X_valid : pandas.DataFrame
        Valid encoded columns.
    """
    skf = StratifiedKFold(n_splits=n_splits, random_state=2019, shuffle=True)

    for train_index, valid_index in skf.split(data, y):
        train_x = data.loc[train_index, columns_names].reset_index(drop=True)
        valid_x = data.loc[valid_index, columns_names].reset_index(drop=True)
        train_y, valid_y = y[train_index], y[valid_index]

        te = TargetEncodingSmoothing(columns_names=columns_names, k=k, f=f)
        X_train = get_CV_target_encoding(train_x, train_y, te, cv=cv_noise)

        te.fit(train_x, train_y)
        X_valid = te.transform(valid_x).values

    return X_train, X_valid
