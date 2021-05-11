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
# ****************************************************

# + colab={"base_uri": "https://localhost:8080/"} id="kI2zqvs_GUWV" executionInfo={"status": "ok", "timestamp": 1620767614705, "user_tz": 300, "elapsed": 20113, "user": {"displayName": "Caleb Anyaeche", "photoUrl": "", "userId": "02380624916791193636"}} outputId="e6e5cbc1-d21c-47f7-d262-657bd1fd6831"
from google.colab import drive 
drive.mount('/content/drive')

# + colab={"base_uri": "https://localhost:8080/"} id="j3t5cYT5QVFv" executionInfo={"status": "ok", "timestamp": 1620767617550, "user_tz": 300, "elapsed": 251, "user": {"displayName": "Caleb Anyaeche", "photoUrl": "", "userId": "02380624916791193636"}} outputId="f73a758b-24a0-4f44-d41d-c0924ffac86e"
# %cd "/content/drive/MyDrive/Colab Notebooks"

# + colab={"base_uri": "https://localhost:8080/"} id="t4Q9V2wfRBAJ" executionInfo={"status": "ok", "timestamp": 1620767620509, "user_tz": 300, "elapsed": 467, "user": {"displayName": "Caleb Anyaeche", "photoUrl": "", "userId": "02380624916791193636"}} outputId="d9c76369-e7de-440d-ab62-1d299f617714"
# %set_env TFDS_DATA_DIR=/content/drive/MyDrive/Colab Notebooks/tfds/

# + id="a7xXX9VE_vDs"
import pandas as pd
import numpy as np

# + id="luyebS0Q_vDy"
df_train = pd.read_csv('./Data/train_amazon.csv')
df_test = pd.read_csv('./Data/test_amazon.csv')

# + id="_Cyk8rnV_vDz" outputId="7d58e45b-f49f-491e-89cb-a5abf3102f31"
df_train.head()

# + id="_2NkaB-2_vD0" outputId="650358d6-ce6b-47c8-cea6-898c7b2edca8"
df_test.head()

# + id="Vq6n1upH_vD0"

