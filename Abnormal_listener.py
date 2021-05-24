#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 24 14:42:49 2021

@author: Zyy Zhou
"""

import tkinter as tk

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
import pyaudio,wave

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


'''
def record_3_sec(name_of_file):
    """PyAudio example: Record a few seconds of audio and save to a WAVE file."""
 
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 8000
    RECORD_SECONDS = 3
    
    p = pyaudio.PyAudio()
    
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    
    print("* recording")
    
    frames = []
    
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
        
        print("* done recording")
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        wf = wave.open("./train/"+name_of_file+".wav", 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()

def record_dataset():
    for number_of_files in range(1000):
        number_of_files_string=str(number_of_files)
        record_3_sec(number_of_files_string)
'''
def start_training():
    #####################################################################
    # generate dataset
    #print("============== GENERATE DATASET ==============")
    information.insert('end',"============== GENERATE DATASET ==============.\n")
    files = functionstore.file_list_generator(cwd)
    train_data = functionstore.list_to_vector_array(files,
                                      msg="generate train_dataset",
                                      n_mels=n_mels,
                                      frames=frames,
                                      n_fft=n_fft,
                                      hop_length=hop_length,
                                      power=power)
    # train model
    #print("================= TRAINING =================")
    information.insert('end',"================= TRAINING =================.\n")
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
    #print("============== TRAINING COMPLETE ==============")
    information.insert('end',"============== TRAINING COMPLETE ==============.\n")

def start_testing():
    #####################################################################
    # read model
    #print('=================LOAD MODEL===================')
    information.insert('end',"=================LOAD MODEL===================.\n")
    model = keras_model.load_model(model_directory)
    model.summary()
    # read audio files in the test folder
    #print('=================READ TEST DATA================')
    information.insert('end',"=================READ TEST DATA================.\n")
    data = functionstore.file_to_vector_array(test_directory,
                                                              n_mels=n_mels,
                                                              frames=frames,
                                                              n_fft=n_fft,
                                                              hop_length=hop_length,
                                                              power=power)
    #calculate anamoly score
    errors = numpy.mean(numpy.square(data - model.predict(data)), axis=1)
    Anomaly_score = numpy.mean(errors)
    #print('=================ANOMALY SCORE================')
    information.insert('end',"=================ANOMALY SCORE================.\n")
    #print(Anomaly_score)
    information.insert('end',Anomaly_score+".\n")

#########################################################################
######################GUI Generation#####################################
#########################################################################
# Creat a root window
root = tk.Tk()
root.title('Abnormal Listener')
#root.iconphoto(True, tk.PhotoImage(file="./pictures/eye.png"))
#root.tk.call('wm', 'iconphoto', root._w, tk.PhotoImage(file="./pictures/eye.png"))
'''
# the height and width of the window
winWidth = 500
winHeight = 400
# get the monitor's resolution
screenWidth = root.winfo_screenwidth()
screenHeight = root.winfo_screenheight()
x = int((screenWidth - winWidth) / 2)
y = int((screenHeight - winHeight) / 2)
# place the window in the center of the monitor
root.geometry("%sx%s+%s+%s" % (winWidth, winHeight, x, y))
'''
# disable size adjustment
root.resizable(0, 0)

# Creat a menubar
menubar = tk.Menu(root)
mainmenu = tk.Menu(menubar, tearoff=0) #a new empty menubar unit
menubar.add_cascade(label='Main', menu=mainmenu)
mainmenu.add_command(label='Clear')
mainmenu.add_command(label='Exit', command=root.quit)
sidemenu = tk.Menu(menubar, tearoff=0) #a new empty menubar unit
menubar.add_cascade(label='Help_V1.0', menu=sidemenu)
sidemenu.add_command(label='User guide')
sidemenu.add_command(label='About')
sidemenu.add_command(label='Acknowledgement')
root.config(menu=menubar)


# creat a title label
title = tk.Label(root, text='ABNORMAL  LISTENER', bg='gray', font=('Arial', 22), width=30, height=2)
title.grid(row=0,columnspan=2)
# Another way to allocate a label:1）l.pack(); 2)l.place();
'''
# Creat a picture window
photo = tk.PhotoImage(file = "./pictures/abnormal_listener_main.png")
picture = tk.Label(root,image=photo)
picture.grid(row=1,columnspan=2,padx=5, pady=5)
'''

# Creat a button for training
train_b = tk.Button(root, text='Start Training', font=('Arial', 18), width=15, height=1,command=start_training())
train_b.grid(row=1,column=0,sticky="w")

# Creat a button for testing
test_b = tk.Button(root, text='Start Monitoring', font=('Arial', 18), width=15, height=1,command=start_testing())
test_b.grid(row=1,column=1,sticky="w")

# Creat a text window
text = tk.Text(root, width=56, height=10)
text.grid(row=2,columnspan=2)

# information bar
information = tk.Label(root,text='GENG5551 Research Project by Zyy Zhou(22670661) 24/05/2021',font=('Arial', 10))
information.grid(row=3,columnspan=2,sticky="e")

# loop the windows
root.mainloop()


#########################################################################
#########################################################################
#########################################################################


