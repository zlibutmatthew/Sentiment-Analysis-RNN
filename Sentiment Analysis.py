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
# -

# 1. Background. What is the importance of this project?
#
# 2. Dataset. Description of your dataset. When the dataset was collected. Who is the author of the dataset? Where did you download it?
#
# 3. Model. Describe your model.
#
# 4. Results. What is the accuracy or other scores you achieved for training sets, testing sets?  What hyperparameters you used (CNN/RNN layers, activation functions, number of nodes in each layer), learning rate, number of epochs etc. Compare results for different hyperparameters. 
#
# 5. Conclusion and future work. Compare your results with others' work for the same or similar datasets. Discuss how can you possibly further improve your results in the future. 

# ### Background
# Sentiment Analysis is a technique used in order to understand the emotional tone behind a series of words. It is used for many different purposes such as gaining an idea of public opinion on certain topics, used to understand customer reviews, for market research, and customer service approaches. There are many nuances in the english language, grammer, slang, cultural variation, and tone can change a review in a way that is hard for a machine to understand. Consider the sentence "My package was delayed. Brilliant!"- most humans would recognize this as sarcasm, but the machine may see "Brilliant!" and decide that this is a positive review. Our goal is to try and maximize the machines accuracy using a Recurrent Neural Network.

# ### Dataset
# Our data was retrieved on Kaggle (https://www.kaggle.com/bittlingmayer/amazonreviews#train.ft.txt.bz2).
# The datasets author is Adam Bittlingmayer
# The dataset can also be found in .csv format in Xiang Zhang's Google Drive directory (https://drive.google.com/drive/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M)

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 430, "status": "ok", "timestamp": 1620776309904, "user": {"displayName": "Caleb Anyaeche", "photoUrl": "", "userId": "02380624916791193636"}, "user_tz": 300} id="kI2zqvs_GUWV" outputId="98212f45-50c6-48e6-f1b5-3282bb21bb24"
using_colab = False 

using_colab = True 
from google.colab import drive 
drive.mount('/content/drive')


# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 373, "status": "ok", "timestamp": 1620776312222, "user": {"displayName": "Caleb Anyaeche", "photoUrl": "", "userId": "02380624916791193636"}, "user_tz": 300} id="j3t5cYT5QVFv" outputId="a7bcfd1a-1c84-4535-b6a6-486ebc8647d3"
# %cd "/content/drive/MyDrive/Colab Notebooks"

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 499, "status": "ok", "timestamp": 1620776314656, "user": {"displayName": "Caleb Anyaeche", "photoUrl": "", "userId": "02380624916791193636"}, "user_tz": 300} id="t4Q9V2wfRBAJ" outputId="32b1a456-783c-4937-c356-c9eaf4196f74"
# %set_env TFDS_DATA_DIR=/content/drive/MyDrive/Colab Notebooks/tfds/

# + executionInfo={"elapsed": 859, "status": "ok", "timestamp": 1620792916376, "user": {"displayName": "Caleb Anyaeche", "photoUrl": "", "userId": "02380624916791193636"}, "user_tz": 300} id="nJEnCR73nX9l"
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

# + colab={"base_uri": "https://localhost:8080/", "height": 204} executionInfo={"elapsed": 1498, "status": "ok", "timestamp": 1620792921669, "user": {"displayName": "Caleb Anyaeche", "photoUrl": "", "userId": "02380624916791193636"}, "user_tz": 300} id="luyebS0Q_vDy" outputId="923986b6-0aff-4430-fc5d-98cdaafbfd98"
# Using training data for the whole dataset. Training data is large enough. 150k instances
if using_colab:
    df = pd.read_csv('./Data/train_amazon.csv')
    #df_test = pd.read_csv('./Data/test_amazon.csv')
else:
    data = pd.read_csv('train_amazon.csv')
    #df_test = pd.read_csv('test_amazon.csv')

df.head()


# + executionInfo={"elapsed": 7548, "status": "ok", "timestamp": 1620792930542, "user": {"displayName": "Caleb Anyaeche", "photoUrl": "", "userId": "02380624916791193636"}, "user_tz": 300} id="xExhCTu39ISA"
#Preprocessing 

def remove_numbers(x):
    x = re.sub('[0-9]{5,}', '#####', x)
    x = re.sub('[0-9]{4}', '####', x)
    x = re.sub('[0-9]{3}', '###', x)
    x = re.sub('[0-9]{2}', '##', x)
    return x

#Removing Numbers
df['text'] = df['text'].apply(lambda x: remove_numbers(x)) 

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 506, "status": "ok", "timestamp": 1620792931908, "user": {"displayName": "Caleb Anyaeche", "photoUrl": "", "userId": "02380624916791193636"}, "user_tz": 300} id="ngXLEK0ll_dM" outputId="ec144c74-d65e-4c53-ae05-59a667a4f5f0"
# Create a dataset

target = df.pop('label')

ds_raw = tf.data.Dataset.from_tensor_slices(
    (df.values, target.values))

## inspection:
for ex in ds_raw.take(3):
    tf.print(ex[0].numpy()[0][:50], ex[1])

# + [markdown] id="0T2tCwX_pJdH"
# TRAIN, VALIDATION, TEST SPLITS

# + executionInfo={"elapsed": 369, "status": "ok", "timestamp": 1620792933971, "user": {"displayName": "Caleb Anyaeche", "photoUrl": "", "userId": "02380624916791193636"}, "user_tz": 300} id="K6HeiEkcpNbm"
tf.random.set_seed(1)

ds_raw = ds_raw.shuffle(
    50000, reshuffle_each_iteration=False)

ds_raw_test = ds_raw.take(25000)
ds_raw_train_valid = ds_raw.skip(25000)
ds_raw_train = ds_raw_train_valid.take(20000)
ds_raw_valid = ds_raw_train_valid.skip(20000)

# + [markdown] id="3afOxC8zppNb"
# TOKENIZER

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 4489, "status": "ok", "timestamp": 1620792940307, "user": {"displayName": "Caleb Anyaeche", "photoUrl": "", "userId": "02380624916791193636"}, "user_tz": 300} id="8fMYEip4puID" outputId="d6ca0a11-b0b3-482c-e1c1-9e4a5ac27a4c"
## find unique tokens (words)

tokenizer = tfds.deprecated.text.Tokenizer()
token_counts = Counter()

for example in ds_raw_train:  
    tokens = tokenizer.tokenize(example[0].numpy()[0])
    token_counts.update(tokens)
    
print('Vocab-size:', len(token_counts))

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 699, "status": "ok", "timestamp": 1620792943727, "user": {"displayName": "Caleb Anyaeche", "photoUrl": "", "userId": "02380624916791193636"}, "user_tz": 300} id="Yd5xlJPXtMod" outputId="f781b1dc-c168-4280-9d6e-7b4cf1a5849a"
## Encoding each unique token into integers

encoder = tfds.deprecated.text.TokenTextEncoder(token_counts)

example_str = 'This is an example!'
encoder.encode(example_str)


# + executionInfo={"elapsed": 459, "status": "ok", "timestamp": 1620792945838, "user": {"displayName": "Caleb Anyaeche", "photoUrl": "", "userId": "02380624916791193636"}, "user_tz": 300} id="N6K115AJtXRp"
## Define the function for transformation

def encode(text_tensor, label):
    text = text_tensor.numpy()[0]
    encoded_text = encoder.encode(text)
    return encoded_text, label

## Wrap the encode function to a TF Op.
def encode_map_fn(text, label):
    return tf.py_function(encode, inp=[text, label], 
                          Tout=(tf.int64, tf.int64))


# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 1480, "status": "ok", "timestamp": 1620792949100, "user": {"displayName": "Caleb Anyaeche", "photoUrl": "", "userId": "02380624916791193636"}, "user_tz": 300} id="s3cbJ2VAth97" outputId="0faae31a-f6e9-4200-9389-e2c6c187317c"
ds_train = ds_raw_train.map(encode_map_fn)
ds_valid = ds_raw_valid.map(encode_map_fn)
ds_test = ds_raw_test.map(encode_map_fn)

tf.random.set_seed(1)
for example in ds_train.shuffle(1000).take(5):
    print('Sequence length:', example[0].shape)

example

# + executionInfo={"elapsed": 408, "status": "ok", "timestamp": 1620792954335, "user": {"displayName": "Caleb Anyaeche", "photoUrl": "", "userId": "02380624916791193636"}, "user_tz": 300} id="EXeYLPn1uOEm"
## batching the datasets
train_data = ds_train.padded_batch(
    32, padded_shapes=([-1],[]))

valid_data = ds_valid.padded_batch(
    32, padded_shapes=([-1],[]))

test_data = ds_test.padded_batch(
    32, padded_shapes=([-1],[]))


# + [markdown] id="E3MnAIbkwND0"
# BUILD RNN MODEL

# + executionInfo={"elapsed": 376, "status": "ok", "timestamp": 1620792959185, "user": {"displayName": "Caleb Anyaeche", "photoUrl": "", "userId": "02380624916791193636"}, "user_tz": 300} id="8Wl6uhEqwPqI"
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



# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 1257, "status": "ok", "timestamp": 1620792962729, "user": {"displayName": "Caleb Anyaeche", "photoUrl": "", "userId": "02380624916791193636"}, "user_tz": 300} id="rejTqqOAxEIn" outputId="3a82b91c-e025-43ec-a73b-ac4203736aa2"
embedding_dim = 20
vocab_size = len(token_counts) + 2

rnn_model = build_rnn_model(
    embedding_dim, vocab_size,
    recurrent_type='SimpleRNN', 
    n_recurrent_units=64,
    n_recurrent_layers=1,
    bidirectional=True)

rnn_model.summary()

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 1586831, "status": "ok", "timestamp": 1620794684906, "user": {"displayName": "Caleb Anyaeche", "photoUrl": "", "userId": "02380624916791193636"}, "user_tz": 300} id="r5FwUWXnynhh" outputId="612ee735-75e7-4389-aea1-cff9cc3a231b"
rnn_model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                  metrics=['accuracy'])


history = rnn_model.fit(
    train_data, 
    validation_data=valid_data, 
    epochs=10)

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 26761, "status": "ok", "timestamp": 1620795431100, "user": {"displayName": "Caleb Anyaeche", "photoUrl": "", "userId": "02380624916791193636"}, "user_tz": 300} id="9FeMYgOZy91f" outputId="9602bdd6-9397-4384-c17f-9bf2c5dac08d"
results = rnn_model.evaluate(test_data)
print('Test Acc.: {:.2f}%'.format(results[1]*100))
# -

# ### Results
# Accuracy for each model
# 1. A Simple RNN and 2 epochs: 84.91%
# 2. A RNN with 1 GRU Layer and 3 epochs: 86.63%
# 3. A RNN with 1 LSTM Layer and 1 epochs: 87.04%
# 4. A RNN with 2 GRU Layer and 5 epochs: 84.48%
# 5. A RNN with 2 LSTM Layer and 1 epochs: 86.36%

# ### Conclusion and Future Work
# The LSTM model trained with 1 epoch was shown to have the best accuracy of 87.04%. If training time weren't an issue, we would be able to train models with much a much more complex layer structure. The future work would to do that. 


