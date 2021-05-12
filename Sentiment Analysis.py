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

# + id="nJEnCR73nX9l" executionInfo={"status": "ok", "timestamp": 1620792916376, "user_tz": 300, "elapsed": 859, "user": {"displayName": "Caleb Anyaeche", "photoUrl": "", "userId": "02380624916791193636"}}
#Import necessary packages 

import re
import tensorflow as tf
from collections import Counter
import numpy as np
import pandas as pd
import tensorflow_datasets as tfds
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Dense

# + id="luyebS0Q_vDy" colab={"base_uri": "https://localhost:8080/", "height": 204} executionInfo={"status": "ok", "timestamp": 1620792921669, "user_tz": 300, "elapsed": 1498, "user": {"displayName": "Caleb Anyaeche", "photoUrl": "", "userId": "02380624916791193636"}} outputId="923986b6-0aff-4430-fc5d-98cdaafbfd98"
# Using training data for the whole dataset. Training data is large enough. 150k instances
if using_colab:
    df = pd.read_csv('./Data/train_amazon.csv')
    #df_test = pd.read_csv('./Data/test_amazon.csv')
else:
    data = pd.read_csv('train_amazon.csv')
    #df_test = pd.read_csv('test_amazon.csv')

df.head()


# + id="xExhCTu39ISA" executionInfo={"status": "ok", "timestamp": 1620792930542, "user_tz": 300, "elapsed": 7548, "user": {"displayName": "Caleb Anyaeche", "photoUrl": "", "userId": "02380624916791193636"}}
#Preprocessing 

def remove_numbers(x):
    x = re.sub('[0-9]{5,}', '#####', x)
    x = re.sub('[0-9]{4}', '####', x)
    x = re.sub('[0-9]{3}', '###', x)
    x = re.sub('[0-9]{2}', '##', x)
    return x

#Removing Numbers
df['text'] = df['text'].apply(lambda x: remove_numbers(x)) 

# + colab={"base_uri": "https://localhost:8080/"} id="ngXLEK0ll_dM" executionInfo={"status": "ok", "timestamp": 1620792931908, "user_tz": 300, "elapsed": 506, "user": {"displayName": "Caleb Anyaeche", "photoUrl": "", "userId": "02380624916791193636"}} outputId="ec144c74-d65e-4c53-ae05-59a667a4f5f0"
# Create a dataset

target = df.pop('label')

ds_raw = tf.data.Dataset.from_tensor_slices(
    (df.values, target.values))

## inspection:
for ex in ds_raw.take(3):
    tf.print(ex[0].numpy()[0][:50], ex[1])

# + [markdown] id="0T2tCwX_pJdH"
# TRAIN, VALIDATION, TEST SPLITS

# + id="K6HeiEkcpNbm" executionInfo={"status": "ok", "timestamp": 1620792933971, "user_tz": 300, "elapsed": 369, "user": {"displayName": "Caleb Anyaeche", "photoUrl": "", "userId": "02380624916791193636"}}
tf.random.set_seed(1)

ds_raw = ds_raw.shuffle(
    50000, reshuffle_each_iteration=False)

ds_raw_test = ds_raw.take(25000)
ds_raw_train_valid = ds_raw.skip(25000)
ds_raw_train = ds_raw_train_valid.take(20000)
ds_raw_valid = ds_raw_train_valid.skip(20000)

# + [markdown] id="3afOxC8zppNb"
# TOKENIZER

# + colab={"base_uri": "https://localhost:8080/"} id="8fMYEip4puID" executionInfo={"status": "ok", "timestamp": 1620792940307, "user_tz": 300, "elapsed": 4489, "user": {"displayName": "Caleb Anyaeche", "photoUrl": "", "userId": "02380624916791193636"}} outputId="d6ca0a11-b0b3-482c-e1c1-9e4a5ac27a4c"
## find unique tokens (words)

tokenizer = tfds.deprecated.text.Tokenizer()
token_counts = Counter()

for example in ds_raw_train:  
    tokens = tokenizer.tokenize(example[0].numpy()[0])
    token_counts.update(tokens)
    
print('Vocab-size:', len(token_counts))

# + colab={"base_uri": "https://localhost:8080/"} id="Yd5xlJPXtMod" executionInfo={"status": "ok", "timestamp": 1620792943727, "user_tz": 300, "elapsed": 699, "user": {"displayName": "Caleb Anyaeche", "photoUrl": "", "userId": "02380624916791193636"}} outputId="f781b1dc-c168-4280-9d6e-7b4cf1a5849a"
## Encoding each unique token into integers

encoder = tfds.deprecated.text.TokenTextEncoder(token_counts)

example_str = 'This is an example!'
encoder.encode(example_str)


# + id="N6K115AJtXRp" executionInfo={"status": "ok", "timestamp": 1620792945838, "user_tz": 300, "elapsed": 459, "user": {"displayName": "Caleb Anyaeche", "photoUrl": "", "userId": "02380624916791193636"}}
## Define the function for transformation

def encode(text_tensor, label):
    text = text_tensor.numpy()[0]
    encoded_text = encoder.encode(text)
    return encoded_text, label

## Wrap the encode function to a TF Op.
def encode_map_fn(text, label):
    return tf.py_function(encode, inp=[text, label], 
                          Tout=(tf.int64, tf.int64))


# + colab={"base_uri": "https://localhost:8080/"} id="s3cbJ2VAth97" executionInfo={"status": "ok", "timestamp": 1620792949100, "user_tz": 300, "elapsed": 1480, "user": {"displayName": "Caleb Anyaeche", "photoUrl": "", "userId": "02380624916791193636"}} outputId="0faae31a-f6e9-4200-9389-e2c6c187317c"
ds_train = ds_raw_train.map(encode_map_fn)
ds_valid = ds_raw_valid.map(encode_map_fn)
ds_test = ds_raw_test.map(encode_map_fn)

tf.random.set_seed(1)
for example in ds_train.shuffle(1000).take(5):
    print('Sequence length:', example[0].shape)

example

# + id="EXeYLPn1uOEm" executionInfo={"status": "ok", "timestamp": 1620792954335, "user_tz": 300, "elapsed": 408, "user": {"displayName": "Caleb Anyaeche", "photoUrl": "", "userId": "02380624916791193636"}}
## batching the datasets
train_data = ds_train.padded_batch(
    32, padded_shapes=([-1],[]))

valid_data = ds_valid.padded_batch(
    32, padded_shapes=([-1],[]))

test_data = ds_test.padded_batch(
    32, padded_shapes=([-1],[]))


# + [markdown] id="E3MnAIbkwND0"
# BUILD RNN MODEL

# + id="8Wl6uhEqwPqI" executionInfo={"status": "ok", "timestamp": 1620792959185, "user_tz": 300, "elapsed": 376, "user": {"displayName": "Caleb Anyaeche", "photoUrl": "", "userId": "02380624916791193636"}}
def build_rnn_model(embedding_dim, vocab_size,
                    recurrent_type='SimpleRNN',
                    n_recurrent_units=64,
                    n_recurrent_layers=1,
                    bidirectional=True):

    tf.random.set_seed(1)

    # build the model
    model = tf.keras.Sequential()
    
    model.add(
        Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            name='embed-layer')
    )
    
    for i in range(n_recurrent_layers):
        return_sequences = (i < n_recurrent_layers-1)
            
        if recurrent_type == 'SimpleRNN':
            recurrent_layer = SimpleRNN(
                units=n_recurrent_units, 
                return_sequences=return_sequences,
                name='simprnn-layer-{}'.format(i))
        elif recurrent_type == 'LSTM':
            recurrent_layer = LSTM(
                units=n_recurrent_units, 
                return_sequences=return_sequences,
                name='lstm-layer-{}'.format(i))
        elif recurrent_type == 'GRU':
            recurrent_layer = GRU(
                units=n_recurrent_units, 
                return_sequences=return_sequences,
                name='gru-layer-{}'.format(i))
        
        if bidirectional:
            recurrent_layer = Bidirectional(
                recurrent_layer, name='bidir-'+recurrent_layer.name)
            
        model.add(recurrent_layer)

    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    
    return model



# + colab={"base_uri": "https://localhost:8080/"} id="rejTqqOAxEIn" executionInfo={"status": "ok", "timestamp": 1620792962729, "user_tz": 300, "elapsed": 1257, "user": {"displayName": "Caleb Anyaeche", "photoUrl": "", "userId": "02380624916791193636"}} outputId="3a82b91c-e025-43ec-a73b-ac4203736aa2"
embedding_dim = 20
vocab_size = len(token_counts) + 2

rnn_model = build_rnn_model(
    embedding_dim, vocab_size,
    recurrent_type='SimpleRNN', 
    n_recurrent_units=64,
    n_recurrent_layers=1,
    bidirectional=True)

rnn_model.summary()

# + colab={"base_uri": "https://localhost:8080/"} id="r5FwUWXnynhh" executionInfo={"status": "ok", "timestamp": 1620794684906, "user_tz": 300, "elapsed": 1586831, "user": {"displayName": "Caleb Anyaeche", "photoUrl": "", "userId": "02380624916791193636"}} outputId="612ee735-75e7-4389-aea1-cff9cc3a231b"
rnn_model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                  metrics=['accuracy'])


history = rnn_model.fit(
    train_data, 
    validation_data=valid_data, 
    epochs=10)

# + colab={"base_uri": "https://localhost:8080/"} id="9FeMYgOZy91f" executionInfo={"status": "ok", "timestamp": 1620795431100, "user_tz": 300, "elapsed": 26761, "user": {"displayName": "Caleb Anyaeche", "photoUrl": "", "userId": "02380624916791193636"}} outputId="9602bdd6-9397-4384-c17f-9bf2c5dac08d"
results = rnn_model.evaluate(test_data)
print('Test Acc.: {:.2f}%'.format(results[1]*100))
