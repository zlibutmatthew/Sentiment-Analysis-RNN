# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import pandas as pd
import numpy as np

df_train = pd.read_csv('train_amazon.csv')
df_test = pd.read_csv('test_amazon.csv')

df_train.head()

df_test.head()


