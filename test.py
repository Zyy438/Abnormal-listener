#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 24 11:30:02 2021

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
# read model
print('=================LOAD MODEL===================')
model = keras_model.load_model(model_directory)
model.summary()
# read audio files in the test folder
print('=================READ TEST DATA================')
data = functionstore.file_to_vector_array(test_directory,
                                                          n_mels=n_mels,
                                                          frames=frames,
                                                          n_fft=n_fft,
                                                          hop_length=hop_length,
                                                          power=power)
#calculate anamoly score
errors = numpy.mean(numpy.square(data - model.predict(data)), axis=1)
Anomaly_score = numpy.mean(errors)
print('=================ANOMALY SCORE================')
print(Anomaly_score)









