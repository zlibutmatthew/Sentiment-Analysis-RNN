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

# + [markdown] id="oKUIypPNr7pb"
# Team Memebers: Caleb Anyaeche & Matthew Zlibut 

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

# + colab={"base_uri": "https://localhost:8080/"} id="kI2zqvs_GUWV" executionInfo={"status": "ok", "timestamp": 1621044136378, "user_tz": 300, "elapsed": 18470, "user": {"displayName": "Caleb Anyaeche", "photoUrl": "", "userId": "02380624916791193636"}} outputId="e3cdc832-2812-4b17-ad7f-6cfc78f66e54"
using_colab = True 
from google.colab import drive 
drive.mount('/content/drive')

#using_colab = False 


# + colab={"base_uri": "https://localhost:8080/"} id="j3t5cYT5QVFv" executionInfo={"status": "ok", "timestamp": 1621044144956, "user_tz": 300, "elapsed": 265, "user": {"displayName": "Caleb Anyaeche", "photoUrl": "", "userId": "02380624916791193636"}} outputId="2359923c-9d75-4a20-9565-dd1e9c0b2077"
# %cd "/content/drive/MyDrive/Colab Notebooks"

# + colab={"base_uri": "https://localhost:8080/"} id="t4Q9V2wfRBAJ" executionInfo={"status": "ok", "timestamp": 1621044146049, "user_tz": 300, "elapsed": 261, "user": {"displayName": "Caleb Anyaeche", "photoUrl": "", "userId": "02380624916791193636"}} outputId="e50a78af-db20-43f1-afce-883d00286a8b"
# %set_env TFDS_DATA_DIR=/content/drive/MyDrive/Colab Notebooks/tfds/

# + id="nJEnCR73nX9l"
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

# + colab={"base_uri": "https://localhost:8080/", "height": 204} id="luyebS0Q_vDy" executionInfo={"status": "ok", "timestamp": 1621044153498, "user_tz": 300, "elapsed": 2587, "user": {"displayName": "Caleb Anyaeche", "photoUrl": "", "userId": "02380624916791193636"}} outputId="8303ed07-b492-4433-f093-844ba6d7584c"
if using_colab:
    df = pd.read_csv('./Data/amazon_reviews.csv')
else:
    data = pd.read_csv('amazon_reviews.csv')

df.head()


# + id="xExhCTu39ISA"
#Preprocessing 

def remove_numbers(x):
    x = re.sub('[0-9]{5,}', '#####', x)
    x = re.sub('[0-9]{4}', '####', x)
    x = re.sub('[0-9]{3}', '###', x)
    x = re.sub('[0-9]{2}', '##', x)
    return x

#Removing Numbers
df['text'] = df['text'].apply(lambda x: remove_numbers(x)) 

# + colab={"base_uri": "https://localhost:8080/"} id="ngXLEK0ll_dM" executionInfo={"status": "ok", "timestamp": 1621044173398, "user_tz": 300, "elapsed": 476, "user": {"displayName": "Caleb Anyaeche", "photoUrl": "", "userId": "02380624916791193636"}} outputId="944510fe-34c9-4d72-a837-3a8b4eaadb39"
# Create a dataset

target = df.pop('label')

ds_raw = tf.data.Dataset.from_tensor_slices(
    (df.values, target.values))

## inspection:
for ex in ds_raw.take(3):
    tf.print(ex[0].numpy()[0][:50], ex[1])

# + [markdown] id="0T2tCwX_pJdH"
# TRAIN, VALIDATION, TEST SPLITS

# + id="K6HeiEkcpNbm"
tf.random.set_seed(1)

ds_raw = ds_raw.shuffle(
    150000, reshuffle_each_iteration=False)

ds_raw_test = ds_raw.take(75000)
ds_raw_train_valid = ds_raw.skip(75000)
ds_raw_train = ds_raw_train_valid.take(37000)
ds_raw_valid = ds_raw_train_valid.skip(37000)

# + [markdown] id="3afOxC8zppNb"
# TOKENIZER

# + colab={"base_uri": "https://localhost:8080/"} id="8fMYEip4puID" executionInfo={"status": "ok", "timestamp": 1621044964227, "user_tz": 300, "elapsed": 7684, "user": {"displayName": "Caleb Anyaeche", "photoUrl": "", "userId": "02380624916791193636"}} outputId="1273666e-79e7-43a7-ea4c-6a48fc6dc6ff"
## find unique tokens (words)

tokenizer = tfds.deprecated.text.Tokenizer()
token_counts = Counter()

for example in ds_raw_train:  
    tokens = tokenizer.tokenize(example[0].numpy()[0])
    token_counts.update(tokens)
    
print('Vocab-size:', len(token_counts))

# + colab={"base_uri": "https://localhost:8080/"} id="Yd5xlJPXtMod" executionInfo={"status": "ok", "timestamp": 1621044968906, "user_tz": 300, "elapsed": 338, "user": {"displayName": "Caleb Anyaeche", "photoUrl": "", "userId": "02380624916791193636"}} outputId="498d9714-7367-42f8-8921-e2b26ac52c53"
## Encoding each unique token into integers

encoder = tfds.deprecated.text.TokenTextEncoder(token_counts)

example_str = 'This is an example!'
encoder.encode(example_str)


# + id="N6K115AJtXRp"
## Define the function for transformation

def encode(text_tensor, label):
    text = text_tensor.numpy()[0]
    encoded_text = encoder.encode(text)
    return encoded_text, label

## Wrap the encode function to a TF Op.
def encode_map_fn(text, label):
    return tf.py_function(encode, inp=[text, label], 
                          Tout=(tf.int64, tf.int64))


# + colab={"base_uri": "https://localhost:8080/"} id="s3cbJ2VAth97" executionInfo={"status": "ok", "timestamp": 1621044982555, "user_tz": 300, "elapsed": 1126, "user": {"displayName": "Caleb Anyaeche", "photoUrl": "", "userId": "02380624916791193636"}} outputId="5517716f-81c7-4d5c-bcfc-23e2444342e8"
ds_train = ds_raw_train.map(encode_map_fn)
ds_valid = ds_raw_valid.map(encode_map_fn)
ds_test = ds_raw_test.map(encode_map_fn)

tf.random.set_seed(1)
for example in ds_train.shuffle(1000).take(5):
    print('Sequence length:', example[0].shape)
#example

# + id="EXeYLPn1uOEm"
## batching the datasets
train_data = ds_train.padded_batch(
    32, padded_shapes=([-1],[]))

valid_data = ds_valid.padded_batch(
    32, padded_shapes=([-1],[]))

test_data = ds_test.padded_batch(
    32, padded_shapes=([-1],[]))


# + [markdown] id="E3MnAIbkwND0"
# BUILD RNN MODEL

# + id="8Wl6uhEqwPqI"
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



# + colab={"base_uri": "https://localhost:8080/"} id="rejTqqOAxEIn" executionInfo={"status": "ok", "timestamp": 1621049774568, "user_tz": 300, "elapsed": 820, "user": {"displayName": "Caleb Anyaeche", "photoUrl": "", "userId": "02380624916791193636"}} outputId="09795024-2045-43c3-9d8e-40ae15d636bf"
embedding_dim = 20
vocab_size = len(token_counts) + 2

rnn_model = build_rnn_model(
    embedding_dim, vocab_size,
    recurrent_type='LSTM', 
    n_recurrent_units=64,
    n_recurrent_layers=1,
    bidirectional=True)

rnn_model.summary()

# + colab={"base_uri": "https://localhost:8080/"} id="r5FwUWXnynhh" executionInfo={"status": "ok", "timestamp": 1621050571346, "user_tz": 300, "elapsed": 435130, "user": {"displayName": "Caleb Anyaeche", "photoUrl": "", "userId": "02380624916791193636"}} outputId="82044a98-96f6-4482-d9a7-19d6548b7b74"
rnn_model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                  metrics=['accuracy'])


history = rnn_model.fit(
    train_data, 
    validation_data=valid_data, 
    epochs=3)

# + id="9FeMYgOZy91f" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1621050952501, "user_tz": 300, "elapsed": 120474, "user": {"displayName": "Caleb Anyaeche", "photoUrl": "", "userId": "02380624916791193636"}} outputId="8b67f471-686f-4f7b-dedf-600bba1fe3c1"
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
#
# NEW ACCURACY FOR EACH MODEL (with corrected data splitting)
# 1. A Simple RNN and 2 epochs: 
# 2. A RNN with 1 GRU Layer and 3 epochs: 87.13
# 3. A RNN with 1 LSTM Layer and 1 epochs: 88.02
# 4. A RNN with 2 GRU Layer and 5 epochs: 
# 5. A RNN with 2 LSTM Layer and 1 epochs: 88.9
# 6. A RNN with 1 GRU Layer and 1 epochs: 88.02
# 7. A RNN with 1 LSTM Layer and 3 epochs: 85.73%

# + [markdown] id="7BZHq45_2083"
# ### Conclusion and Future Work
# The LSTM model trained with 1 epoch was shown to have the best accuracy of 87.04%. If training time weren't an issue, we would be able to train models with much a much more complex layer structure. Also, getting more data on amazon reviews reviews would help since we're working with text. The future work would be to do those. 

# + id="rJzXkY9C2083"

