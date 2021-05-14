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
# To run this code in **Google Colab**.
#
# Google Drive Structure:
#
# - My Drive 
#   - Colab Notebooks
#     - Data
#       - amazon_reviews.csv
#     - Sentiment Analysis.ipynb
#     
# Uncomment "using_colab = False" on line 4 to run this code on jupyter.
#
# ****************************************************

# + [markdown] id="EpFGOWLv208v"
# # Sentiment Analysis on Amazon Commerce Reviews - TensorFlow

# + [markdown] id="A-Odp34Y208w"
# ### Background
# Sentiment Analysis is a technique used in order to understand the emotional tone behind a series of words. It is used for many different purposes such as gaining an idea of public opinion on certain topics, used to understand customer reviews, for market research, and customer service approaches. There are many nuances in the english language, grammer, slang, cultural variation, and tone can change a review in a way that is hard for a machine to understand. Consider the sentence "My package was delayed. Brilliant!"- most humans would recognize this as sarcasm, but the machine may see "Brilliant!" and decide that this is a positive review. Our goal is to try and maximize the machines accuracy using a Recurrent Neural Network.

# + [markdown] id="dwcrUhzG208x"
# ### Dataset
# Our dataset was a subset of the data retrieved on Kaggle (https://www.kaggle.com/bittlingmayer/amazonreviews#train.ft.txt.bz2).
# The datasets author is Adam Bittlingmayer
# The dataset can also be found in .csv format in Xiang Zhang's Google Drive directory (https://drive.google.com/drive/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M)
#
# Dataset has 150k instances with 2 attributes 

# + [markdown] id="PKxuVzPx208x"
# #### Variables
# Independent - Text (string of sentences). 
# Dependent -Label (Negative 0, Positive 1)

# + colab={"base_uri": "https://localhost:8080/"} id="kI2zqvs_GUWV" executionInfo={"status": "ok", "timestamp": 1620962020438, "user_tz": 300, "elapsed": 208, "user": {"displayName": "Caleb Anyaeche", "photoUrl": "", "userId": "02380624916791193636"}} outputId="a3be9b4e-4f7b-4523-ff21-e98e9ba36546"
using_colab = True 
from google.colab import drive 
drive.mount('/content/drive')

#using_colab = False 


# + colab={"base_uri": "https://localhost:8080/"} id="j3t5cYT5QVFv" executionInfo={"status": "ok", "timestamp": 1620962022584, "user_tz": 300, "elapsed": 197, "user": {"displayName": "Caleb Anyaeche", "photoUrl": "", "userId": "02380624916791193636"}} outputId="7e3c120d-f4e3-44ee-824e-e34b96a6bdc2"
# %cd "/content/drive/MyDrive/Colab Notebooks"

# + colab={"base_uri": "https://localhost:8080/"} id="t4Q9V2wfRBAJ" executionInfo={"status": "ok", "timestamp": 1620962024376, "user_tz": 300, "elapsed": 213, "user": {"displayName": "Caleb Anyaeche", "photoUrl": "", "userId": "02380624916791193636"}} outputId="5fb3db8a-ec90-4b7e-d99c-15933853a8ac"
# %set_env TFDS_DATA_DIR=/content/drive/MyDrive/Colab Notebooks/tfds/

# + id="nJEnCR73nX9l" executionInfo={"status": "ok", "timestamp": 1620962025801, "user_tz": 300, "elapsed": 174, "user": {"displayName": "Caleb Anyaeche", "photoUrl": "", "userId": "02380624916791193636"}}
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

# + colab={"base_uri": "https://localhost:8080/", "height": 204} id="luyebS0Q_vDy" executionInfo={"status": "ok", "timestamp": 1620962028315, "user_tz": 300, "elapsed": 1032, "user": {"displayName": "Caleb Anyaeche", "photoUrl": "", "userId": "02380624916791193636"}} outputId="47f25c0f-cc94-40e1-e8ca-38fbe8706234"
if using_colab:
    df = pd.read_csv('./Data/amazon_reviews.csv')
else:
    data = pd.read_csv('amazon_reviews.csv')

df.head()


# + id="xExhCTu39ISA" executionInfo={"status": "ok", "timestamp": 1620962040121, "user_tz": 300, "elapsed": 7143, "user": {"displayName": "Caleb Anyaeche", "photoUrl": "", "userId": "02380624916791193636"}}
#Preprocessing 

def remove_numbers(x):
    x = re.sub('[0-9]{5,}', '#####', x)
    x = re.sub('[0-9]{4}', '####', x)
    x = re.sub('[0-9]{3}', '###', x)
    x = re.sub('[0-9]{2}', '##', x)
    return x

#Removing Numbers
df['text'] = df['text'].apply(lambda x: remove_numbers(x)) 

# + colab={"base_uri": "https://localhost:8080/"} id="ngXLEK0ll_dM" executionInfo={"status": "ok", "timestamp": 1620962049249, "user_tz": 300, "elapsed": 660, "user": {"displayName": "Caleb Anyaeche", "photoUrl": "", "userId": "02380624916791193636"}} outputId="b9739c1a-8600-46b0-c711-31bb8541cb39"
# Create a dataset

target = df.pop('label')

ds_raw = tf.data.Dataset.from_tensor_slices(
    (df.values, target.values))

## inspection:
for ex in ds_raw.take(3):
    tf.print(ex[0].numpy()[0][:50], ex[1])

# + [markdown] id="0T2tCwX_pJdH"
# TRAIN, VALIDATION, TEST SPLITS

# + id="K6HeiEkcpNbm" executionInfo={"status": "ok", "timestamp": 1620962056994, "user_tz": 300, "elapsed": 207, "user": {"displayName": "Caleb Anyaeche", "photoUrl": "", "userId": "02380624916791193636"}}
tf.random.set_seed(1)

ds_raw = ds_raw.shuffle(
    50000, reshuffle_each_iteration=False)

ds_raw_test = ds_raw.take(25000)
ds_raw_train_valid = ds_raw.skip(25000)
ds_raw_train = ds_raw_train_valid.take(20000)
ds_raw_valid = ds_raw_train_valid.skip(20000)

# + [markdown] id="3afOxC8zppNb"
# TOKENIZER

# + colab={"base_uri": "https://localhost:8080/"} id="8fMYEip4puID" executionInfo={"status": "ok", "timestamp": 1620962063586, "user_tz": 300, "elapsed": 4208, "user": {"displayName": "Caleb Anyaeche", "photoUrl": "", "userId": "02380624916791193636"}} outputId="d035093c-f6cf-4e1a-b993-74cabb5ea9cf"
## find unique tokens (words)

tokenizer = tfds.deprecated.text.Tokenizer()
token_counts = Counter()

for example in ds_raw_train:  
    tokens = tokenizer.tokenize(example[0].numpy()[0])
    token_counts.update(tokens)
    
print('Vocab-size:', len(token_counts))

# + colab={"base_uri": "https://localhost:8080/"} id="Yd5xlJPXtMod" executionInfo={"status": "ok", "timestamp": 1620962065746, "user_tz": 300, "elapsed": 302, "user": {"displayName": "Caleb Anyaeche", "photoUrl": "", "userId": "02380624916791193636"}} outputId="43737e96-62e1-4af0-ed04-58872ebe5c2e"
## Encoding each unique token into integers

encoder = tfds.deprecated.text.TokenTextEncoder(token_counts)

example_str = 'This is an example!'
encoder.encode(example_str)


# + id="N6K115AJtXRp" executionInfo={"status": "ok", "timestamp": 1620962071784, "user_tz": 300, "elapsed": 203, "user": {"displayName": "Caleb Anyaeche", "photoUrl": "", "userId": "02380624916791193636"}}
## Define the function for transformation

def encode(text_tensor, label):
    text = text_tensor.numpy()[0]
    encoded_text = encoder.encode(text)
    return encoded_text, label

## Wrap the encode function to a TF Op.
def encode_map_fn(text, label):
    return tf.py_function(encode, inp=[text, label], 
                          Tout=(tf.int64, tf.int64))


# + colab={"base_uri": "https://localhost:8080/"} id="s3cbJ2VAth97" executionInfo={"status": "ok", "timestamp": 1620962074059, "user_tz": 300, "elapsed": 1051, "user": {"displayName": "Caleb Anyaeche", "photoUrl": "", "userId": "02380624916791193636"}} outputId="f72a748a-e28a-4516-df53-ab643ada01f0"
ds_train = ds_raw_train.map(encode_map_fn)
ds_valid = ds_raw_valid.map(encode_map_fn)
ds_test = ds_raw_test.map(encode_map_fn)

tf.random.set_seed(1)
for example in ds_train.shuffle(1000).take(5):
    print('Sequence length:', example[0].shape)

example

# + id="EXeYLPn1uOEm" executionInfo={"status": "ok", "timestamp": 1620962078446, "user_tz": 300, "elapsed": 236, "user": {"displayName": "Caleb Anyaeche", "photoUrl": "", "userId": "02380624916791193636"}}
## batching the datasets
train_data = ds_train.padded_batch(
    32, padded_shapes=([-1],[]))

valid_data = ds_valid.padded_batch(
    32, padded_shapes=([-1],[]))

test_data = ds_test.padded_batch(
    32, padded_shapes=([-1],[]))


# + [markdown] id="E3MnAIbkwND0"
# BUILD RNN MODEL

# + id="8Wl6uhEqwPqI" executionInfo={"status": "ok", "timestamp": 1620962081971, "user_tz": 300, "elapsed": 198, "user": {"displayName": "Caleb Anyaeche", "photoUrl": "", "userId": "02380624916791193636"}}
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



# + colab={"base_uri": "https://localhost:8080/"} id="rejTqqOAxEIn" executionInfo={"status": "ok", "timestamp": 1620962085988, "user_tz": 300, "elapsed": 453, "user": {"displayName": "Caleb Anyaeche", "photoUrl": "", "userId": "02380624916791193636"}} outputId="7a9139e8-01dc-468f-c191-9d00db3f6918"
embedding_dim = 20
vocab_size = len(token_counts) + 2

rnn_model = build_rnn_model(
    embedding_dim, vocab_size,
    recurrent_type='SimpleRNN', 
    n_recurrent_units=64,
    n_recurrent_layers=1,
    bidirectional=True)

rnn_model.summary()

# + colab={"base_uri": "https://localhost:8080/", "height": 392} id="r5FwUWXnynhh" executionInfo={"status": "error", "timestamp": 1620962116340, "user_tz": 300, "elapsed": 15700, "user": {"displayName": "Caleb Anyaeche", "photoUrl": "", "userId": "02380624916791193636"}} outputId="939b423c-3608-4f6b-a37d-7c7c94a5d4c9"
rnn_model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                  metrics=['accuracy'])


history = rnn_model.fit(
    train_data, 
    validation_data=valid_data, 
    epochs=10)

# + id="9FeMYgOZy91f"
results = rnn_model.evaluate(test_data)
print('Test Acc.: {:.2f}%'.format(results[1]*100))

# + [markdown] id="twRP7CEv2083"
# ### Results
# Accuracy for each model
# 1. A Simple RNN and 2 epochs: 84.91%
# 2. A RNN with 1 GRU Layer and 3 epochs: 86.63%
# 3. A RNN with 1 LSTM Layer and 1 epochs: 87.04%
# 4. A RNN with 2 GRU Layer and 5 epochs: 84.48%
# 5. A RNN with 2 LSTM Layer and 1 epochs: 86.36%

# + [markdown] id="7BZHq45_2083"
# ### Conclusion and Future Work
# The LSTM model trained with 1 epoch was shown to have the best accuracy of 87.04%. If training time weren't an issue, we would be able to train models with much a much more complex layer structure. Also, getting more data on amazon reviews reviews would help since we're working with text. The future work would be to do those. 

# + id="rJzXkY9C2083"

