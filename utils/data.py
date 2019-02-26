import pickle

def pickleDumpChunks(df, path, split_size=3, inplace=False):
    """
    path = '../output/mydf'

    wirte '../output/mydf/0.p'
          '../output/mydf/1.p'
          '../output/mydf/2.p'

    """
    if inplace==True:
        df.reset_index(drop=True, inplace=True)
    else:
        df = df.reset_index(drop=True)
    mkdir(path)

    for i in tqdm(range(split_size)):
        df.ix[df.index%split_size==i].to_pickle(path+'/{}.p'.format(i))

    return

def pickleLoadChunks(path, col=None):
    if col is None:
        df = pd.concat([pd.read_pickle(f) for f in tqdm(sorted(glob(path+'/*')))])
    else:
        df = pd.concat([pd.read_pickle(f)[col] for f in tqdm(sorted(glob(path+'/*')))])
    return df

def csvToPickle(path):
    '''Convert csv to pickle format

        Parameters
        ----------
        path: str
            filepath
    '''
    data = pd.read_csv(path)
    joblib.dump(data, path[:-4]+'.p')
