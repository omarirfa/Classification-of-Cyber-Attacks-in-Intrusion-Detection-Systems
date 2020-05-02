import pandas as pd
import numpy as np
# a = sys.argv[1]
# data = pd.read_csv("a")
data = pd.read_csv("mixedset.csv")
# filtered_data = data.replace('',np.nan)
filtered_data = data.dropna(axis="columns", how="any")
filtered_data.to_csv('cleanednormalizedmixedset.csv')
