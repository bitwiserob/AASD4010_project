# path/filename: main.py

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed
from tensorflow.keras.layers import Input, Dense, Dropout
# from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import os
import shutil


def preprocess_data():
    nasdaq100_data = pd.read_pickle('../data/data_pkls/nasdaq100.pkl')
    # Getting only 'Adj Close' column for each company

    df = pd.DataFrame()

    # Loop through each company
    for company in nasdaq100_data.columns.levels[0]:
        company_data = nasdaq100_data[company]

        # Calculate daily returns, volatility, and average return for each company
        daily_returns = company_data['Adj Close'].pct_change()
        volatility = daily_returns.rolling(window=20).std()
        # average_return = daily_returns.rolling(window=20).mean()

        company_features = pd.DataFrame({
            f'{company}': volatility,
            # f'{company}_AverageReturn': average_return
        })

        df = pd.concat([df, company_features], axis=1)
    return df['2010':'2020']


def clean_data(df):
    # Dropping completely empty columns
    empty_columns = df.columns[df.isnull().all()]
    df_cleaned = df.drop(empty_columns, axis=1)
    # Filling NaN values with mean
    imputer = SimpleImputer(strategy='mean')
    imputer.fit(df_cleaned)
    df_cleaned = imputer.transform(df_cleaned)   
    
    return df_cleaned, empty_columns


    
def scale_data(df_cleaned):
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df_cleaned)
    return df_scaled

def build_autoencoder(df_scaled):
    input_dim = df_scaled.shape[1]  # The number of columns in the training data

    encoding_dim = 2  # Set the desired dimensionality of the embeddings

    # Reshape the input data correctly for LSTM layers
    input_data = df_scaled.reshape(df_scaled.shape[0], 1, df_scaled.shape[1])

    # Print the shape of the reshaped data
    print(input_data.shape)

    # Encoder
    encoder_inputs = Input(shape=(None, input_dim))
    encoder = LSTM(100, return_sequences=True)(encoder_inputs)
    encoder = LSTM(encoding_dim)(encoder)

    # Decoder
    decoder_inputs = RepeatVector(input_data.shape[1])(encoder)  # Using the time step size of input_data
    decoder = LSTM(100, return_sequences=True)(decoder_inputs)
    decoder_outputs = TimeDistributed(Dense(input_dim, activation='tanh'))(decoder)

    # Autoencoder model
    autoencoder = Model(inputs=encoder_inputs, outputs=decoder_outputs)

    # Define an optimizer
    optimizer = tf.keras.optimizers.Adam()  # Using Adam optimizer with default settings

    autoencoder.compile(optimizer=optimizer, loss='mse')
    autoencoder.fit(input_data, input_data, epochs=50, batch_size=32, shuffle=True)

    return autoencoder

def cluster_and_visualize(autoencoder, input_data):
    # Extracting embeddings from the trained autoencoder
    encoder = tf.keras.Model(inputs=autoencoder.input, outputs=autoencoder.layers[1].output)
    embeddings = encoder.predict(input_data)

    # Reshaping embeddings
    embeddings_reshaped = embeddings.reshape((embeddings.shape[0], -1))

    # Apply K-means clustering to the embeddings
    num_clusters = 3
    kmeans = KMeans(n_init=10, n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(embeddings_reshaped)

    # Define cluster colors
    cluster_colors = {0: 'red', 1: 'blue', 2: 'green'}

    # Plotting
    scatter = plt.scatter(embeddings_reshaped[:, 0], embeddings_reshaped[:, 1], c=clusters, cmap='tab10', edgecolors='black')
    plt.xlabel('Embedding Dimension 1')
    plt.ylabel('Embedding Dimension 2')
    plt.title('Clustering of NASDAQ 100 Stocks using Autoencoder')

    plt.legend(handles=scatter.legend_elements()[0], labels=cluster_colors.keys())

    plt.show()
    
def extract_features(nasdaq_data):
    df = pd.DataFrame()
    for company in nasdaq_data.columns.levels[0]:
        company_data = nasdaq_data[company]
        daily_returns = company_data['Adj Close'].pct_change()
        volatility = daily_returns.rolling(window=20).std()
        # Uncomment for average return
        # average_return = daily_returns.rolling(window=20).mean()

        company_features = pd.DataFrame({
            f'{company}': volatility,  # Replace with average_return if needed
        })

        df = pd.concat([df, company_features], axis=1)
    return df['2010':'2020']

def build_and_train_autoencoder(df_scaled):
    input_dim = df_scaled.shape[1]  # Corrected to reflect the number of features
    encoding_dim = 2

    input_data = df_scaled.reshape(df_scaled.shape[0], df_scaled.shape[1], 1)

    optimizer = Adam(learning_rate=0.001)

    encoder_inputs = Input(shape=(None, input_dim))
    encoder = LSTM(100, return_sequences=True)(encoder_inputs)
    encoder = LSTM(encoding_dim)(encoder)

    decoder_inputs = RepeatVector(input_dim)(encoder)
    decoder = LSTM(100, return_sequences=True)(decoder_inputs)
    decoder_outputs = TimeDistributed(Dense(input_dim, activation='tanh'))(decoder)

    autoencoder = Model(inputs=encoder_inputs, outputs=decoder_outputs)
    autoencoder.compile(optimizer=optimizer, loss='mse')
    autoencoder.fit(input_data, input_data, epochs=50, batch_size=32, shuffle=True)

    return autoencoder



def extract_embeddings_and_cluster(autoencoder, input_data):
    # Extract embeddings from the trained autoencoder
    
    encoder = tf.keras.Model(inputs=autoencoder.input, outputs=autoencoder.layers[1].output)

    embeddings = encoder.predict(input_data)

    # Reshape embeddings for clustering
    embeddings_reshaped = embeddings.reshape((embeddings.shape[0], -1))

    # K-means clustering
    num_clusters = 3
    kmeans = KMeans(n_init=10, n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(embeddings_reshaped)

    return embeddings_reshaped, clusters

def visualize_clusters(embeddings_reshaped, clusters):
    # Define cluster colors
    cluster_colors = {0: 'red', 1: 'blue', 2: 'green'}

    # Scatter plot for clusters
    scatter = plt.scatter(embeddings_reshaped[:, 0], embeddings_reshaped[:, 1], c=clusters, cmap='tab10', edgecolors='black')
    plt.xlabel('Embedding Dimension 1')
    plt.ylabel('Embedding Dimension 2')
    plt.title('Clustering of NASDAQ 100 Stocks using Autoencoder')

    plt.legend(handles=scatter.legend_elements()[0], labels=cluster_colors.keys())

    plt.show()

if __name__ == "__main__":
    # Load the NASDAQ 100 data
    nasdaq100_data = pd.read_pickle('../data/data_pkls/nasdaq100.pkl')

    # Extract features from the NASDAQ data
    df = extract_features(nasdaq100_data)

    # Clean the extracted data
    df_cleaned, empty_columns = clean_data(df)

    # Scale the cleaned data
    df_scaled = scale_data(df_cleaned)

    # Build and train the autoencoder model with the scaled data
    autoencoder = build_and_train_autoencoder(df_scaled)

    # Extract embeddings and perform clustering
    embeddings_reshaped, clusters = extract_embeddings_and_cluster(autoencoder, df_scaled)

    # Visualize the clusters
    visualize_clusters(embeddings_reshaped, clusters)