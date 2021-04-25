# Project set up 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('USvideos.csv')
data.head()

print('There are', str(len(data)), 'rows in this dataset')

data.info()
