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

# + [markdown] id="zZOwGUpCUi5m"
# ****************************************************
#
# Run this code in **Google Colab**.
#
# Google Drive Structure:
#
# - My Drive 
#   - Colab Notebooks
#     - Data
#       - train_amazon.csv
#       - test_amazon.csv
#     - Sentiment Analysis.ipynb
#     
# Uncomment the code below to run on colab.
#
# ****************************************************

# + colab={"base_uri": "https://localhost:8080/"} id="kI2zqvs_GUWV" executionInfo={"status": "ok", "timestamp": 1620776309904, "user_tz": 300, "elapsed": 430, "user": {"displayName": "Caleb Anyaeche", "photoUrl": "", "userId": "02380624916791193636"}} outputId="98212f45-50c6-48e6-f1b5-3282bb21bb24"
using_colab = False 

using_colab = True 
from google.colab import drive 
drive.mount('/content/drive')


# + colab={"base_uri": "https://localhost:8080/"} id="j3t5cYT5QVFv" executionInfo={"status": "ok", "timestamp": 1620776312222, "user_tz": 300, "elapsed": 373, "user": {"displayName": "Caleb Anyaeche", "photoUrl": "", "userId": "02380624916791193636"}} outputId="a7bcfd1a-1c84-4535-b6a6-486ebc8647d3"
# %cd "/content/drive/MyDrive/Colab Notebooks"

# + colab={"base_uri": "https://localhost:8080/"} id="t4Q9V2wfRBAJ" executionInfo={"status": "ok", "timestamp": 1620776314656, "user_tz": 300, "elapsed": 499, "user": {"displayName": "Caleb Anyaeche", "photoUrl": "", "userId": "02380624916791193636"}} outputId="32b1a456-783c-4937-c356-c9eaf4196f74"
# %set_env TFDS_DATA_DIR=/content/drive/MyDrive/Colab Notebooks/tfds/

# + id="nJEnCR73nX9l" executionInfo={"status": "ok", "timestamp": 1620790975000, "user_tz": 300, "elapsed": 497, "user": {"displayName": "Caleb Anyaeche", "photoUrl": "", "userId": "02380624916791193636"}}
#Import necessary packages 

import re
import tensorflow as tf
from collections import Counter
import numpy as np
import pandas as pd
import tensorflow_datasets as tfds

# + id="luyebS0Q_vDy" colab={"base_uri": "https://localhost:8080/", "height": 204} executionInfo={"status": "ok", "timestamp": 1620790978025, "user_tz": 300, "elapsed": 1226, "user": {"displayName": "Caleb Anyaeche", "photoUrl": "", "userId": "02380624916791193636"}} outputId="abfa1902-18b3-45d1-b815-012f12019fb6"
# Using training data for the whole dataset. Training data is large enough. 150k instances
if using_colab:
    df = pd.read_csv('./Data/train_amazon.csv')
    #df_test = pd.read_csv('./Data/test_amazon.csv')
else:
    data = pd.read_csv('train_amazon.csv')
    #df_test = pd.read_csv('test_amazon.csv')

df.head()


# + id="xExhCTu39ISA" executionInfo={"status": "ok", "timestamp": 1620790990722, "user_tz": 300, "elapsed": 7329, "user": {"displayName": "Caleb Anyaeche", "photoUrl": "", "userId": "02380624916791193636"}}
#Preprocessing 

def remove_numbers(x):
    x = re.sub('[0-9]{5,}', '#####', x)
    x = re.sub('[0-9]{4}', '####', x)
    x = re.sub('[0-9]{3}', '###', x)
    x = re.sub('[0-9]{2}', '##', x)
    return x

#Removing Numbers
df['text'] = df['text'].apply(lambda x: remove_numbers(x)) 

# + colab={"base_uri": "https://localhost:8080/"} id="ngXLEK0ll_dM" executionInfo={"status": "ok", "timestamp": 1620790992983, "user_tz": 300, "elapsed": 380, "user": {"displayName": "Caleb Anyaeche", "photoUrl": "", "userId": "02380624916791193636"}} outputId="46ff6946-b58f-4262-9c05-d2b543e8893e"
# Create a dataset

target = df.pop('label')

ds_raw = tf.data.Dataset.from_tensor_slices(
    (df.values, target.values))

## inspection:
for ex in ds_raw.take(3):
    tf.print(ex[0].numpy()[0][:50], ex[1])

# + [markdown] id="0T2tCwX_pJdH"
# TRAIN, VALIDATION, TEST SPLITS

# + id="K6HeiEkcpNbm" executionInfo={"status": "ok", "timestamp": 1620790996550, "user_tz": 300, "elapsed": 438, "user": {"displayName": "Caleb Anyaeche", "photoUrl": "", "userId": "02380624916791193636"}}
tf.random.set_seed(1)

ds_raw = ds_raw.shuffle(
    50000, reshuffle_each_iteration=False)

ds_raw_test = ds_raw.take(25000)
ds_raw_train_valid = ds_raw.skip(25000)
ds_raw_train = ds_raw_train_valid.take(20000)
ds_raw_valid = ds_raw_train_valid.skip(20000)

# + [markdown] id="3afOxC8zppNb"
# TOKENIZER

# + colab={"base_uri": "https://localhost:8080/"} id="8fMYEip4puID" executionInfo={"status": "ok", "timestamp": 1620791003114, "user_tz": 300, "elapsed": 4193, "user": {"displayName": "Caleb Anyaeche", "photoUrl": "", "userId": "02380624916791193636"}} outputId="67b7c683-88dd-4cd3-ddbc-af2d4f2cdc66"
## find unique tokens (words)

tokenizer = tfds.deprecated.text.Tokenizer()
token_counts = Counter()

for example in ds_raw_train:  
    tokens = tokenizer.tokenize(example[0].numpy()[0])
    token_counts.update(tokens)
    
print('Vocab-size:', len(token_counts))
