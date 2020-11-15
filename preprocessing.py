from memory_profiler import profile
# In[1]:
import os
import librosa
from pydub import AudioSegment
import numpy as np
from random import sample
from tqdm import tqdm
import numpy as np
import random
import re
import threading
from math import ceil

feature_names = [
    'filename',
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


# In[23]:

FILENAME="data.csv"

csv = open(FILENAME, "w")
csv.write(",".join(map(str, feature_names)) + "\n")
csv.close()

print("INIT")

# In[2]:

# get_ipython().system('ls audios/')
# get_ipython().system('pwd')


# In[3]:


# WAV_PATH = "/home/fstovarr/birds/test"
WAV_PATH = "/home/fstovarr/birds/audios/wav"
#MP3_PATH = "/home/fstovarr/birds/audios/mp3"


# In[5]:


wav_files = os.listdir(WAV_PATH)
# mp3_files = os.listdir(MP3_PATH)

SAMPLES = len(wav_files)
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

wav_files = wav_files[0:SAMPLES]

# In[10]:

def process_data(processed_data, files):
    x, sr = (None, None)
    class_name = ""
    amplitudes = []
    np_fft = []
    for file in tqdm(files):
        try:
            x, sr = librosa.load("{}/{}".format(WAV_PATH, file), sr=None)
        
            class_name = re.search('([A-za-z-]+)-[0-9]+.wav', file).group(1).lower()

            np_fft = np.fft.fft(x)
            amplitudes = 2/len(x) * np.abs(np_fft)
            processed_data[0] = file
            processed_data[1] = class_name

            i = 2
            for f in [x,amplitudes,spectral_centroid(x, sr),  spectral_rolloff(x, sr),  spectral_bandwidth(x, sr),  mel_frequency(x, sr),  chroma(x, sr)]:
                processed_data[i] = np.mean(f)
                processed_data[i+1] = np.max(f)
                processed_data[i+2] = np.min(f)
                processed_data[i+3] = np.std(f)
                i += 4

            processed_data[i] = (sum(zero_crossing_rate(x, sr)))

            csv = open(FILENAME, "a+")
            csv.write(",".join(map(str, processed_data[0:i+1])) + "\n")
            csv.close()
        except Exception as e:
            print(e, "ERROR WITH FILE {}".format(file))
            continue

def start():
    threads = THREADS
    chunk_size = ceil(len(wav_files) / threads)
    jobs = []

    out_list = [[''] * 31 for i in range(threads)]

    for i in range(0, threads):
        thread = threading.Thread(target=process_data, args=(out_list[i], wav_files[chunk_size * i : min(chunk_size * i + chunk_size, len(wav_files))]))
        jobs.append(thread)

    print("THREADS {}".format(THREADS))

    for j in jobs:
        print("THREAD")
        j.start()

    for j in jobs:
        j.join()

    print("FINISH")
    
start()
