#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa


# Создаю фрейм для наших данных

# In[ ]:


train = pd.read_csv('../new-data-without_silence/train_gt.csv', header=None, names=['name', 'label'])


# Я создал списки по среднеквадратичному и отклонению

# In[ ]:


class Transformer():

    def __init__(self) -> None:
        self.sample_rate = 22050 
        self.FRAME_SIZE = 256 
        self.HOP_LENGTH = 64 
        self.FRAME_SIZE_FFT = 512 
        self.HOP_SIZE_FFT = 128

    def amplitude_envelope(self, audio, frame_size, hop_length):
        return np.array([max(audio[i:i+frame_size]) for i in range(0, audio.size, hop_length)])

    def mfccs_ob(self, audio, sample_rate):
        mfccs = librosa.feature.mfcc(y=audio, n_mfcc=13, sr=sample_rate)
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfcc = librosa.feature.delta(mfccs, order=2)
        mfccs_audio = np.concatenate((mfccs, delta_mfccs, delta2_mfcc))
        return mfccs_audio

    def obrabotka(self, audio_path):
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        amplitude_list.append(self.amplitude_envelope(audio, self.FRAME_SIZE, self.HOP_LENGTH))
        mfccs_list.append(self.mfccs_ob(audio, self.sample_rate))
        rms_list.append(librosa.feature.rms(y=audio, frame_length=self.FRAME_SIZE, hop_length=self.HOP_LENGTH)[0])
        zcr_list.append(librosa.feature.zero_crossing_rate(y=audio, frame_length=self.FRAME_SIZE, hop_length=self.HOP_LENGTH)[0])
        stft_list.append(np.abs(librosa.stft(audio, n_fft=self.FRAME_SIZE_FFT, hop_length=self.HOP_SIZE_FFT)))
        cent_list.append(librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate))
        down_list.append(librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate))

tr = Transformer()


# In[ ]:


amplitude_list = []
mfccs_list = []
rms_list = []
zcr_list = []
stft_list = []
cent_list = []
down_list = []


# In[ ]:


get_ipython().run_line_magic('time', '')
from tqdm.autonotebook import tqdm

for i in tqdm(train['name']):
    tr.obrabotka(f'../new-data-without_silence/train/{i}')


# In[ ]:


d = {
    'amplitude': list(map(lambda x: x.reshape(-1, 1),amplitude_list)),
    'mfccs': list(map(lambda x: x.reshape(-1, 39),mfccs_list)), 
    'rms': list(map(lambda x: x.reshape(-1, 1),rms_list)),
    'zcr': list(map(lambda x: x.reshape(-1, 1),zcr_list)),
    'stft': list(map(lambda x: x.reshape(-1, 257),stft_list)),
    'cent': list(map(lambda x: x.reshape(-1, 1),cent_list)),
    'down': list(map(lambda x: x.reshape(-1, 1),down_list)),
}
shapes = {
    'amplitude': (None, 1),
    'mfccs': (None, 39), 
    'rms': (None, 1),
    'zcr': (None, 1),
    'stft': (None, 257),
    'cent': (None, 1),
    'down': (None, 1),
}
types = {c: np.float32 for c in shapes.keys()}


# In[ ]:


import tensorflow as tf

def create_cnn(model_name: str, input_shape: tuple):
    try:
        input = tf.keras.layers.Input(shape=input_shape)

        x = tf.keras.layers.Conv1D(8, kernel_size=3, activation='relu')(input)
        x = tf.keras.layers.Conv1D(16, 3, activation='relu')(x)
        x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
        x = tf.keras.layers.Dropout(0.25)(x)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        
        model = tf.keras.Model(inputs=input, outputs=x)

        return model
    except:
        print(model_name)


# In[ ]:


models = {
    name: create_cnn(name, shape) for name, shape in shapes.items()
}


# In[ ]:


inputs = {name: tf.keras.layers.Input(shape=shape, name=name) for name, shape in shapes.items()}
outputs = [model(inputs[name]) for name, model in models.items()]

y = tf.keras.layers.Concatenate(axis=1)(outputs)
y = tf.keras.layers.Dense(1, activation='sigmoid')(y)

model = tf.keras.Model(inputs=inputs, outputs=y)


# In[ ]:


model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer='adam', metrics=['f1_score'])


# In[ ]:


model.summary()


# In[ ]:


for data in d.values():
    print(len(data))


# In[ ]:


model.fit(list(d.values()), train['label'], epochs=10, batch_size=32)


# In[ ]:


tf.keras.config.disable_traceback_filtering()

