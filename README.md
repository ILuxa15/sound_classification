# Sound Classification with Convolutional Neural Network
A TensorFlow convolutional neural network model for sound classification.

### Dependencies
* TensorFlow
* NumPy
* Librosa
* scikit-learn

### Feature Extraction and Neural Network Model
Project contains two modules — `feature_processing.py` and `cnn_model.py`.
* `features_processing.py` allow to extract spectrogram or melspectrogram from audio file. Also you can set duration of audio
or split it to equivalent length segment and extract feature from each.
* `cnn_model.py` has implementation of neural network with Keras module from TensorFlow.

### Example
As an example in this project was perform training of model using [UrbanSound8K](https://zenodo.org/record/1203745#.XVQVe0dn1hF)
dataset.
#### Training
* For training you should modify `example_train.py` to have valid values of following arguments:
  * `dataset_path`: path where dataset is stored;
  * `features_train_save_path`, `labels_train_save_path`: path where train part of dataset will be saved;
  * `features_test_save_path`, `labels_test_save_path`: path where test part of dataset will be saved;
  * `scalers_save_path`: path where data for scaling will be saved.
* Then just run it `.\example_train.py`.
#### Evaluation
* For evaluate data or predict class of audio file modify following arguments from `example_predict.py`:
  * `model_path`: path to the pre-trained model;
  * `features_path`, `labels_path`: path to the data which will be evaluated;
  * `scalers_path`: path to the data for scaling;
  * `sample_path`: path to the audio file whose class will be predicted.
* And run `.\example_predict.py`.
