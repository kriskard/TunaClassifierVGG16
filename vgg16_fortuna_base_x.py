import keras,os
import tensorflow
from tensorflow.keras.applications import VGG16 
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

trdata = ImageDataGenerator(rescale=1./255)
traindata = trdata.flow_from_directory(directory="dataset/train",target_size=(224,224),batch_size=32,class_mode='categorical')
valdata = ImageDataGenerator(rescale=1./255)
validdata = valdata.flow_from_directory(directory="dataset/valid", target_size=(224,224),batch_size=32,class_mode='categorical')

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3)) 

model = Sequential() 
model.add(base_model) 
model.add(Flatten()) 
model.add(Dense(512, activation='relu'))
model.add(Dense(3, activation='softmax'))

from keras.optimizers import Adam
opt = Adam(lr=0.001)
model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

model.summary()

from keras.callbacks import ModelCheckpoint, EarlyStopping
checkpoint = ModelCheckpoint("vgg16tuna_base.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=20, verbose=1, mode='auto')
hist = model.fit_generator(x=traindata,steps_per_epoch=100,validation_data=validdata, validation_steps=10,epochs=100,callbacks=[checkpoint,early])

import matplotlib.pyplot as plt
plt.plot(hist.history["acc"])
plt.plot(hist.history['val_acc'])
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title("model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Accuracy","Validation Accuracy","loss","Validation Loss"])
plt.show()