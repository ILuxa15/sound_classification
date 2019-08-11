import os
import csv
import joblib
import numpy as np
import tensorflow as tf
from feature_processing import extract_feature
from cnn_model import get_model
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# TO DO
# Need as command line arguments
features = []
labels = []
dataset_path = '/home/ilya/Documents/UrbanSound8K/audio'
features_train_save_path = '/home/ilya/Documents/GitHub/sound_classification/dataset/us8k_features_train.npy'
features_test_save_path = '/home/ilya/Documents/GitHub/sound_classification/dataset/us8k_features_test.npy'
labels_train_save_path = '/home/ilya/Documents/GitHub/sound_classification/dataset/us8k_labels_train.npy'
labels_test_save_path = '/home/ilya/Documents/GitHub/sound_classification/dataset/us8k_labels_test.npy'
scalers_save_path = '/home/ilya/Documents/GitHub/sound_classification/dataset/us8k_scales.pkl'

# Dataset processing
for folder in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder)

    for audio in os.listdir(folder_path):
        audio_path = os.path.join(folder_path, audio)

        # Extract melspectrogram from an audiofile with 4 seconds duration
        feature = extract_feature(audio_path, 'melspec', duration=4)
        features.append(feature)

        # A digit after first hyphen (-) is a class id
        label = audio.rsplit('-')[1]
        labels.append(label)

features = np.array(features)
labels = np.array(labels)

# Split dataset into train and test sets
features_train, features_test, labels_train, labels_test = train_test_split(
    features, labels, test_size=0.3)

# Save datasets
np.save(features_train_save_path, features_train)
np.save(features_test_save_path, features_test)
np.save(labels_train_save_path, labels_train)
np.save(labels_test_save_path, labels_test)

# # If you've already extracted features then load files
# features_train = np.load(features_train_save_path)
# features_test = np.load(features_test_save_path)
# labels_train = np.load(labels_train_save_path)
# labels_test = np.load(labels_test_save_path)

# Normalizing features by StandartScaler from scikit-learn and save scalers
scalers = {}
for idx in range(features_train.shape[1]):
    scalers[idx] = StandardScaler()
    scalers[idx].fit(features_train[:, idx, :])
    features_train[:, idx, :] = scalers[idx].transform(
        features_train[:, idx, :])

for idx in range(features_test.shape[1]):
    features_test[:, idx, :] = scalers[idx].transform(features_test[:, idx, :])

joblib.dump(scalers, scalers_save_path)

# One hot encode labels sets
encoder = OneHotEncoder(sparse=False)
labels_train = encoder.fit_transform(
    labels_train.reshape(len(labels_train), 1))
labels_test = encoder.fit_transform(labels_test.reshape(len(labels_test), 1))

# Transpose feature matrices to 4D shape (samples, rows, columns, channels)
features_train = np.transpose(
    np.array(features_train, ndmin=4), axes=(1, 2, 3, 0))
features_test = np.transpose(
    np.array(features_test, ndmin=4), axes=(1, 2, 3, 0))

# Init callbacks for a model
earlystop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=3, verbose=0, mode='auto')
save_model = tf.keras.callbacks.ModelCheckpoint(
    'model-{epoch:03d}-{acc:03f}-{val_acc:03f}.h5', save_best_only=True,
    monitor='val_loss')

# Init a model and start training
model = get_model(features_train.shape[1:4], labels_train.shape[1])
model.fit(features_train, labels_train, epochs=500, callbacks=[
          earlystop, save_model], validation_data=(features_test, labels_test))
