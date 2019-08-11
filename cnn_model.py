import tensorflow as tf


def get_model(input_shape, output_len):
    """Return model of convolutional neural network for sound classification


    Parameters
    ----------
    input_shape : tuple
        shape of input data

    output_len : int > 1
        number of outputs

    Returns
    -------
    model : tf.keras.Sequential
        Model of neural network

    """

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(24, 5, data_format='channels_last',
                                     padding='same', activation='relu',
                                     input_shape=input_shape))
    model.add(tf.keras.layers.MaxPool2D(2))

    model.add(tf.keras.layers.Conv2D(
        48, 5, activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPool2D(2))

    model.add(tf.keras.layers.Conv2D(
        48, 5, activation='relu', padding='same'))

    model.add(tf.keras.layers.MaxPool2D(2))

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(64, activation='relu'))

    model.add(tf.keras.layers.Dense(
        output_len, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  metrics=['accuracy'], optimizer='SGD')

    return model
