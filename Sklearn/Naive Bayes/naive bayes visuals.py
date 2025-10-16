# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 10:37:36 2023

@author: Omega Joctan
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Change this directory to be the one containing the Files

directory = r"C:\Users\Omega Joctan\AppData\Roaming\MetaQuotes\Terminal\F4F6C6D7A7155578A6DEA66D12B1D40D\MQL5\Files\NAIVE BAYES"

data = pd.read_csv(f"{directory}\\vars.csv")

for var in data:
    sns.distplot(data[var])
    plt.show()