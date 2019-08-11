import os
import librosa
import numpy as np
import audioread.exceptions


def extract_spectrogram(y, to_db=True, **kwargs):
    """Extract spectrogram using Short-time Fourier transform (STFT) from `librosa.core.stft()`


    Parameters
    ----------
    y : np.ndarray [shape=(n,)]
        the input signal (audio time series)

    to_db : boolean
        convert amplitude values to dB-scaled values

    kwargs : additional keyword arguments
        Additional arguments for `librosa.core.stft()`

    Returns
    -------
    spec : np.ndarray [shape=(1 + n_fft/2, t), dtype=dtype]
        STFT matrix

    """

    spec = librosa.stft(y, **kwargs)

    if to_db:
        spec = librosa.amplitude_to_db(np.abs(spec), ref=np.max)

    return spec


def extract_melspectrogram(y=None, sr=22050, to_db=True, **kwargs):
    """Extract mel-scaled spectrogram using `librosa.feature.melspectrogram()`


    Parameters
    ----------
    y : np.ndarray [shape=(n,)] or None
        the input signal (audio time series)

    sr : number > 0 [scalar]
        sampling rate of `y`

    to_db : boolean
        convert power values to dB-scaled values

    kwargs : additional keyword arguments
        Additional arguments for `librosa.feature.melspectrogram()`

    Returns
    -------
    melspec : np.ndarray [shape=(n_mels, t)]
        Mel spectrogram matrix

    """

    melspec = librosa.feature.melspectrogram(y, **kwargs)

    if to_db:
        melspec = librosa.power_to_db(melspec, ref=np.max)

    return melspec


def extract_feature(path, feature_type='spec', sr=22050, mono=True, duration=None,
                    split=False, **kwargs):
    """Extract feature of certain type from audio file`


    Parameters
    ----------
    path : string
        path to the audio file

    feature_type : string
        feature type to extract

    sr : number
        target sampling rate

    mono : boolean
        convert audio to mono

    duration : float
        only load up to this much audio (in seconds)

    split : boolean or None
        - If `True` split audio file into equivalent segments and extract feature
        from each
        - If 'False' extract feature from beginning to `duration` of audio file
        - If None extract feature from whole file
        Default is None.

    kwargs : additional keyword arguments
        Additional arguments for `librosa.feature` methods

    Returns
    -------
    feature : np.ndarray
        Feature matrix or array of feature matrices

    """

    # Choose extraction method depending on feature type
    if feature_type == 'spec':
        get_feature = extract_spectrogram
    elif feature_type == 'melspec':
        get_feature = extract_melspectrogram
    else:
        raise ValueError('Can\'t recognize feature type')

    try:
        y, sr = librosa.load(path, sr=sr, mono=mono)
    except audioread.exceptions.NoBackendError:
        print('Audio file of unknown format - ' + path)
        return

    if duration is None:
        # Return feature from whole audio file
        return get_feature(y, sr, **kwargs)

    # Calculate number of samples
    desired_length = duration * sr
    audio_length = int(np.floor(librosa.get_duration(filename=path) * sr))

    if split:
        # Split audio file into equivalent segments and extract feature from each
        features = []

        pos = 0
        while pos + desired_length <= audio_length:
            feature = get_feature(y[pos:pos + desired_length], sr, **kwargs)
            features.append(feature)
            pos += desired_length

        if pos < audio_length:
            # Zero padding to duration
            y = librosa.util.fix_length(y[pos:audio_length], desired_length)
            feature = get_feature(y, sr, **kwargs)
            features.append(feature)

        return np.array(features)
    else:
        # Otherwise extract feature from file with duration parameter length
        if audio_length < desired_length:
            y = librosa.util.fix_length(y, desired_length)
        elif audio_length > desired_length:
            y = y[0:desired_length]

        feature = get_feature(y, sr, **kwargs)

        return feature