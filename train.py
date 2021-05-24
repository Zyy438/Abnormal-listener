#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 23 21:24:01 2021
The abnormal sound detection (Training a model from folder 'train')
@author: Zyy Zhou
"""

import os
cwd=os.getcwd()
print('Current working directory:'+cwd)
import glob
import sys
import numpy
import keras_model
import librosa
import librosa.core
import librosa.feature
import functionstore

#define parameters
#dataset_directory=cwd+'/train'
model_directory=cwd+'/model'
test_directory=cwd+'/test/testfile.wav'
#result_directory=cwd+'/result'
max_fpr=0.1
n_mels=128
frames=5
n_fft=1024
hop_length=512
power=2.0
epochs=100
batch_size=512
shuffle=True
validation_split=0.1
verbose=1

#####################################################################
# generate dataset
print("============== GENERATE DATASET ==============")
files = functionstore.file_list_generator(cwd)
train_data = functionstore.list_to_vector_array(files,
                                  msg="generate train_dataset",
                                  n_mels=n_mels,
                                  frames=frames,
                                  n_fft=n_fft,
                                  hop_length=hop_length,
                                  power=power)
# train model
print("================= TRAINING =================")
model = keras_model.get_model(n_mels * frames)
model.summary()
model.compile(loss='mean_squared_error',optimizer='adam')
history = model.fit(train_data,
                    train_data,
                    epochs=epochs,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    validation_split=validation_split,
                    verbose=verbose)
model.save(model_directory)
print("============== TRAINING COMPLETE ==============")

