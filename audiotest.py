import os
import librosa
import numpy as np
import json
import soundfile as sf
import tensorflow as tf
import keras


def extract_mfcc(filename):
  y,sr=librosa.load(filename,duration=3,offset=0.5)
  mfcc=np.mean(librosa.feature.mfcc(y=y,sr=sr,n_mfcc=40).T,axis=0)
  return mfcc

def audio_files(filename):

  target_length = 44100
  audio, sr = librosa.load(filename, sr=None)
  threshold = 0.02 * np.max(np.abs(audio))  # Adjust the threshold as needed
  audio = np.where(np.abs(audio) > threshold, audio, 0)
  if len(audio) > target_length:
    audio = audio[:target_length]
  else:
    padding = target_length - len(audio)
    audio = np.pad(audio, (0, padding), mode='constant')
  return audio

def extract_chroma(filename):
  y,sr=librosa.load(filename,duration=3,offset=0.5)
  chroma=np.mean(librosa.feature.chroma_stft(y=y, sr=sr, hop_length=512).T,axis=0)
  return chroma

from keras.models import model_from_json
file=open("models\path_to_saved_model(2).json", 'r')
loaded_model=file.read()
file.close()
audio_model=model_from_json(loaded_model)

weight_files = [
    "models\path_to_saved_weights(1).h5",
    "models\path_to_saved_weights(2).h5",
    "models\path_to_saved_weights(3).h5"
]

for weight_file in weight_files:
    audio_model.load_weights(weight_file)

def remove_noise_and_save(x_raw_audio, sample_rate, output_filename):
    # Compute the Short-Time Fourier Transform (STFT)
    stft = librosa.stft(x_raw_audio)

    # Compute the magnitude of the STFT
    magnitude = np.abs(stft)

    # Compute the mean magnitude across time frames
    mean_magnitude = np.mean(magnitude, axis=1, keepdims=True)

    # Apply spectral subtraction: subtract the mean magnitude from the original magnitude
    denoised_magnitude = magnitude - mean_magnitude

    # Reconstruct the denoised audio using the modified magnitude and the original phase
    denoised_stft = denoised_magnitude * np.exp(1j * np.angle(stft))

    # Inverse STFT to get the denoised audio
    denoised_audio = librosa.istft(denoised_stft)

    # Save the denoised audio to a WAV file
    sf.write(output_filename, denoised_audio, sample_rate)



def detect_multiple_voices(audio_data, sample_rate):
    # Ensure the audio data is mono
    if len(audio_data.shape) != 1:
        raise ValueError("Input audio data should be mono")

denoised_output_filename = "/content/1038_ITS_HAP_XX_noise.wav"
# Define a function to load audio data
def load_audio(filename, target_length=44100):
    audio_data, sample_rate = librosa.load(filename, sr=None, mono=True)  # Load as mono

    # Resample the audio to the target length if needed
    if len(audio_data) != target_length:
        audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=target_length)
    return audio_data

def preprocess_input_data(mfcc_features, chroma_features, raw_audio_data):
    # Normalize the input data if needed
    mfcc_features_normalized = mfcc_features / np.max(mfcc_features)
    chroma_features_normalized = chroma_features / np.max(chroma_features)
    raw_audio_normalized = raw_audio_data / np.max(raw_audio_data)

    # Reshape the features for the model input
    mfcc_features_reshaped = mfcc_features_normalized.reshape(1, -1, 1)
    chroma_features_reshaped = chroma_features_normalized.reshape(1, -1, 1)
    raw_audio_reshaped = raw_audio_normalized.reshape(1, -1, 1)

    return mfcc_features_reshaped, chroma_features_reshaped, raw_audio_reshaped

def predict_with_model(model, mfcc_features, chroma_features, raw_audio_data):
    # Preprocess the input data
    mfcc_features_reshaped, chroma_features_reshaped, raw_audio_reshaped = preprocess_input_data(
        mfcc_features, chroma_features, raw_audio_data
    )

    # Make predictions
    predictions = model.predict([mfcc_features_reshaped, chroma_features_reshaped, raw_audio_reshaped])

    # Get the predicted class
    predicted_class = np.argmax(predictions, axis=1)

    return predicted_class

def predict(filename):
    target_length = 44100
    x_raw_audio = load_audio(filename, target_length)
    audio_data, sample_rate = librosa.load(filename, sr=None, mono=True)  # Load as mono
    result = detect_multiple_voices(audio_data, sample_rate)
    if result:
        print("Warning: Multiple voices detected")
    else:
        print("Single voice detected")
    target_length = 44100
    x_raw_audio=audio_files(filename)
    x_audio=np.array(x_raw_audio)
    x_audio=np.expand_dims(x_audio,-1)
    
    mfcc=extract_mfcc(filename)
    mfcc_features=np.array(mfcc)
    mfcc_features=np.expand_dims(mfcc_features,-1)
    
    chroma=extract_chroma(filename)
    chroma_features=np.array(chroma)
    chroma_features=np.expand_dims(chroma,-1)
    
    predicted_class = predict_with_model(audio_model, mfcc_features, chroma_features, x_audio)
    print("Predicted class:", predicted_class)

    if predicted_class == 1 or predicted_class == 3:
        print("Confident")
    else:
        print("Unconfident")


