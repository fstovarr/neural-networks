#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import librosa
from pydub import AudioSegment
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
from random import sample
from tqdm import tqdm
import numpy as np
import pandas as pd
import random
import re


# In[2]:

# get_ipython().system('ls audios/')
# get_ipython().system('pwd')


# In[3]:


WAV_PATH = "/home/fstovarr/birds/audios/wav"
MP3_PATH = "/home/fstovarr/birds/audios/mp3"


# In[5]:


wav_files = os.listdir(WAV_PATH)
mp3_files = os.listdir(MP3_PATH)

SAMPLES = 16
THREADS = 8


# In[6]:


def espectrogram(x):
    X = librosa.stft(x)
    Xdb = librosa.amplitude_to_db(abs(X))
    return Xdb

def spectral_centroid(x, sr):
    return librosa.feature.spectral_centroid(x, sr=sr)[0]

def spectral_rolloff(x, sr):
    return librosa.feature.spectral_rolloff(x+0.01, sr=sr)[0]

def spectral_bandwidth(x, sr):
    spectral_bandwidth_2 = librosa.feature.spectral_bandwidth(x+0.01, sr=sr)[0]
    spectral_bandwidth_3 = librosa.feature.spectral_bandwidth(x+0.01, sr=sr, p=3)[0]
    spectral_bandwidth_4 = librosa.feature.spectral_bandwidth(x+0.01, sr=sr, p=4)[0]
    return (spectral_bandwidth_2, spectral_bandwidth_3, spectral_bandwidth_4)

def zero_crossing_rate(x, sr, n0 = 9000, n1 = 9100):
    return librosa.zero_crossings(x[n0:n1], pad=False)

def mel_frequency(x, sr):
    return librosa.feature.mfcc(x, sr=sr)

def chroma(x, sr):
    return librosa.feature.chroma_stft(x, sr=sr)


# In[7]:


random.shuffle(wav_files)
wav_files


# In[8]:


wav_files = wav_files[0:SAMPLES]


# In[9]:


import threading
from math import ceil


# In[10]:


def process_data(processed_data, files):
    print(files)
    for file in tqdm(files):
        try:
            x, sr = librosa.load("{}/{}".format(WAV_PATH, file), sr=None)
        except Exception as e:
            print(e, "ERROR WITH FILE {}".format(file))
            continue
        
        tmp = []

        class_name = re.search('([A-za-z-]+)-[0-9]+.wav', file).group(1).lower()

        np_fft = np.fft.fft(x)
        amplitudes = 2/len(x) * np.abs(np_fft)
        tmp.append(file)
        tmp.append(class_name)

        features = [
            x,
            amplitudes,
            spectral_centroid(x, sr),  
            spectral_rolloff(x, sr),  
            spectral_bandwidth(x, sr),  
            mel_frequency(x, sr),  
            chroma(x, sr)
        ]

        for f in features:
            tmp.append(np.mean(f))
            tmp.append(np.max(f))
            tmp.append(np.min(f))
            tmp.append(np.std(f))

        tmp.append(sum(zero_crossing_rate(x, sr)))

        processed_data.append(tmp)

threads = THREADS
chunk_size = ceil(len(wav_files) / threads)
jobs = []

out_list = [[]] * threads

for i in range(0, threads):
    print(out_list)
    chunk = np.array(wav_files[chunk_size * i : min(chunk_size * i + chunk_size, len(wav_files))])    
    thread = threading.Thread(
        target=process_data, 
        args=(
            out_list[i], 
            chunk
        )
    )
    jobs.append(thread)
    
for j in jobs:
    j.start()

for j in jobs:
    j.join()


# In[18]:

ol = np.array(out_list)
sz = ol.shape
processed_data = ol.reshape((sz[0] * sz[1], sz[2]))


# In[19]:


feature_names = [
    'file_name',
    'class_name', 
    'wave_min', 
    'wave_max', 
    'wave_mean', 
    'wave_var', 
    'amp_min', 
    'amp_max', 
    'amp_mean', 
    'amp_var', 
    'spectral_centroid_min', 
    'spectral_centroid_max', 
    'spectral_centroid_mean', 
    'spectral_centroid_var', 
    'spectral_rolloff_min', 
    'spectral_rolloff_max', 
    'spectral_rolloff_mean', 
    'spectral_rolloff_var', 
    'spectral_bandwidth_min', 
    'spectral_bandwidth_max', 
    'spectral_bandwidth_mean', 
    'spectral_bandwidth_var', 
    'mel_frequency_min', 
    'mel_frequency_max', 
    'mel_frequency_mean', 
    'mel_frequency_var', 
    'chroma_min', 
    'chroma_max', 
    'chroma_mean', 
    'chroma_var',
    'zero_crossing_rate'
 ]


# In[20]:


data = pd.DataFrame(data=processed_data, columns=feature_names)
data


# In[21]:


data.describe()


# In[22]:


data.to_csv('first{}.csv'.format(SAMPLES))