# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 02:36:32 2020

@author: mukmc
"""

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D,Dropout,MaxPooling2D,Dense,GlobalAveragePooling2D,Flatten
from tensorflow.keras.models import Sequential


from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd

dat = pd.read_csv("Data/data.csv",index_col=False)

newdat = dat[{'bmi','name'}].copy()

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
target_size=(64,64),
drop_duplicates=False,
validate_filenames=True)

val_generator=datagen.flow_from_dataframe(dataframe=newdat,
directory="Data/Images",
x_col="name",
y_col="bmi",
subset="validation",
batch_size=32,
seed=42,
shuffle=True,
class_mode="raw",
target_size=(64,64),
drop_duplicates=False,
validate_filenames=True)


model = Sequential()
model.add(Conv2D(filters=16,kernel_size=(5,5),strides=1,input_shape=(64,64,3),activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=32,kernel_size=(3,3),strides=1,activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(units=512,activation="relu"))
model.add(Dense(units=128,activation="relu"))
model.add(Dense(units=1))
model.compile(optimizer = 'adam', loss = 'mse')

history=model.fit_generator(
        train_generator,
        steps_per_epoch=(3365/32),
        epochs=20,
        validation_data=val_generator,
        validation_steps=(841/32))



model.evaluate()