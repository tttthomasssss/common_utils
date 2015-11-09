# coding: utf-8
from discoutils.thesaurus_loader import Vectors
import pandas as pd
get_ipython().magic('cd ../common_utils/')
df = pd.read_csv('test/test.h5', header=0, index_col=0)
v = Vectors.from_pandas_df(df)
v.init_sims(n_neighbors=5)
print(v.get_nearest_neighbours('issues'))