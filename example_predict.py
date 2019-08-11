import os
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
features_path = '/home/ilya/Documents/GitHub/sound_classification/dataset/us8k_features_test.npy'
labels_path = '/home/ilya/Documents/GitHub/sound_classification/dataset/us8k_labels_test.npy'
scalers_path = '/home/ilya/Documents/GitHub/sound_classification/dataset/us8k_scales.pkl'
sample_path = '/home/ilya/Documents/UrbanSound8K/audio/fold1/24074-1-0-1.wav'
model_path = '/home/ilya/Documents/GitHub/sound_classification/model-009-0.892507-0.869084.h5'

# Load a model
model = tf.keras.models.load_model(model_path)

# Let's check test data
features = np.load(features_path)
labels = np.load(labels_path)
scalers = joblib.load(scalers_path)

for idx in range(features.shape[1]):
    features[:, idx, :] = scalers[idx].transform(features[:, idx, :])
features = np.transpose(np.array(features, ndmin=4), axes=(1, 2, 3, 0))

encoder = OneHotEncoder(sparse=False)
labels = encoder.fit_transform(labels.reshape(len(labels), 1))

model.evaluate(features, labels)

# Let's predict on sample
sample_feature = extract_feature(sample_path, 'melspec', duration=4)
sample_feature = np.array(sample_feature, ndmin=3)
for idx in range(sample_feature.shape[1]):
    sample_feature[:, idx, :] = scalers[idx].transform(
        sample_feature[:, idx, :])
sample_feature = np.transpose(
    np.array(sample_feature, ndmin=4), axes=(1, 2, 3, 0))
result = model.predict(sample_feature)

# Print predicted class name
classes = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark',
           'drilling', 'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music']
print(classes[np.argmax(result[0])])
