# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 02:06:33 2020

@author: mukmc
"""
import tensorflow as tf

from tensorflow.keras.models import load_model

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
model = load_model('Pretrained_Models/facenet_keras.h5')

i=0
for l in model.layers:
    if(i>420):
        break
    else:
        model.layers[i].trainable=False
    i=i+1

newmod=Sequential()
newmod.add(model)
newmod.add(Dense(units=1))
newmod.compile(optimizer = 'adam', loss = 'mean_squared_error')

import pandas as pd

dat = pd.read_csv('Data/data.csv',index_col=False)

newdat = dat[{'bmi','name'}].copy()

from tensorflow.keras.preprocessing.image import ImageDataGenerator


datagen=ImageDataGenerator(rescale=1./255.,validation_split=0.2)

train_generator=datagen.flow_from_dataframe(dataframe=newdat,
directory="Data/Images",
x_col="name",
y_col="bmi",
subset="training",
batch_size=32,
seed=42,
shuffle=True,
class_mode="raw",
target_size=(160,160),
drop_duplicates=False,
validate_filenames=True)

val_generator=datagen.flow_from_dataframe(dataframe=newdat,
directory="Images",
x_col="name",
y_col="bmi",
subset="validation",
batch_size=32,
seed=42,
shuffle=True,
class_mode="raw",
target_size=(160,160),
drop_duplicates=False,
validate_filenames=True)

history=newmod.fit_generator(
        train_generator,
        steps_per_epoch=(3365/32),
        epochs=20,
        validation_data=val_generator,
        validation_steps=(841/32))

