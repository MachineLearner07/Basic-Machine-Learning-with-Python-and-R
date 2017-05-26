# -*- coding: utf-8 -*-
"""
Created on Fri May 26 20:13:09 2017

@author: Hasib Iqbal
"""

# Importing Library

import numpy as np
import pandas as pd
import matplotlib.pyplot as plot


# Importing Dataset

dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,3].values
