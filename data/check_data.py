from pandas import (
    DataFrame, HDFStore
)
import pandas as pd
import numpy as np

#store = HDFStore('metr-la.h5')
df = pd.read_hdf('metr-la.h5')

#print(store)
#print(type(store))
print(df)
print(df.shape)

