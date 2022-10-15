import tensorflow as tf
import librosa
import numpy as np
import pandas as pd

model = tf.keras.models.load_model('model')


def single_extract_features(file_name):
    X, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
    stft = np.abs(librosa.stft(X))
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)
    label = 0
    return mfccs, chroma, mel, contrast, tonnetz, label

def instant_predict(file):
    features_labels = single_extract_features(file)
    features = []
    features.append(np.concatenate((features_labels[0], features_labels[1], features_labels[2], features_labels[3], features_labels[4]), axis=0))
    X = np.array(features)
    # new_model = tf.keras.models.load_model('drive/MyDrive/Voice Detection/Saved Models/catdogmodel1')
    val = model.predict(X)
    print(f"Prediction:{val}")
    preds = np.argmax(val, axis=1)
    print(f"Argmax:{preds}")
    return preds
