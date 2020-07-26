# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 03:41:13 2020

@author: mukmc
"""


import tensorflow as tf

from tensorflow.keras.models import load_model

from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.models import Sequential
model = load_model('Pretrained_Models/facenet_keras.h5')

i=0
for l in model.layers:
    #if(i>400):
     #   break
    #else:
    model.layers[i].trainable=False
    i=i+1

newmod=Sequential()
newmod.add(model)
newmod.add(Dense(units=256,activation='relu'))
newmod.add(Dropout(0.1))
newmod.add(Dense(units=4,activation="softmax"))
newmod.compile(optimizer = 'adam', loss = 'categorical_crossentropy',metrics=['accuracy'])

import pandas as pd


dat = pd.read_csv('Data/data.csv',index_col=False)

newdat = dat[{'bmi','name'}].copy()

yo=[]
for i in newdat['bmi']:
    if(i >=30.0):
        yo.append("obese")
    elif(i <18.5):
        yo.append("underweight")
    elif(i>=25.0 and i<30):
        yo.append("overweight")
    else:
        yo.append("normal")
newdat['class']=yo

from sklearn.model_selection import train_test_split

train , test = train_test_split(newdat,test_size=0.1,shuffle=True,random_state=27)


from tensorflow.keras.preprocessing.image import ImageDataGenerator


datagen=ImageDataGenerator(rescale=1./255.,validation_split=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

train_generator=datagen.flow_from_dataframe(dataframe=train,
directory="Data/Images",
x_col="name",
y_col="class",
subset="training",
batch_size=32,
seed=27,
shuffle=False,
class_mode="categorical",
target_size=(160,160),
drop_duplicates=False,
validate_filenames=True)

val_generator=datagen.flow_from_dataframe(dataframe=train,
directory="Data/Images",
x_col="name",
y_col="class",
subset="validation",
batch_size=32,
seed=27,
shuffle=False,
class_mode="categorical",
target_size=(160,160),
drop_duplicates=False,
validate_filenames=True)


datagen2=ImageDataGenerator(rescale=1./255.)

test_generator=datagen2.flow_from_dataframe(dataframe=test,
directory="Data/Images",
x_col="name",
y_col="class",
batch_size=32,
seed=27,
shuffle=False,
class_mode="categorical",
target_size=(160,160),
drop_duplicates=False,
validate_filenames=True)

history=newmod.fit_generator(
        train_generator,
        steps_per_epoch=(3028/32),
        epochs=50,
        validation_data=val_generator,
        validation_steps=(757/32))

newmod.evaluate_generator(test_generator)
