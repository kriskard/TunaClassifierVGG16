import tensorflow as tf
from tensorflow.keras.applications import VGG16 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# Data generators
trdata = ImageDataGenerator(
    rescale=1./255,               # Rescale pixel values (0-255) to [0, 1]
    shear_range=0.2,              # Shear transformation
    zoom_range=0.2,               # Random zoom
    horizontal_flip=True          # Flip images horizontally
)
traindata = trdata.flow_from_directory(directory="C:/VGG16/tuna_classification/dataset/train", target_size=(224, 224), batch_size=32, class_mode='categorical',shuffle=False)

valdata = ImageDataGenerator(rescale=1./255)
validdata = valdata.flow_from_directory(directory="C:/VGG16/tuna_classification/dataset/valid", target_size=(224, 224), batch_size=32, class_mode='categorical')

# Base model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3)) 

# Model setup
model = Sequential() 
model.add(base_model) 
model.add(Flatten()) 
model.add(Dense(512, activation='relu'))
model.add(Dense(3, activation='softmax'))

# Compile model
opt = Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# Amount of data and summary
print(f"Total training images: {traindata.samples}")
print(f"Total validation images: {validdata.samples}")
model.summary()

# Callbacks
checkpoint = ModelCheckpoint("vgg16tuna_base.keras", monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto')
early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=10, verbose=1, mode='auto')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-5, verbose=1)

# Fit model
hist = model.fit(traindata, steps_per_epoch=35, validation_data=validdata, validation_steps=9, epochs=30, callbacks=[checkpoint, early, reduce_lr])

# Plotting
plt.plot(hist.history["accuracy"])
plt.plot(hist.history['val_accuracy'])
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Accuracy", "Validation Accuracy", "Loss", "Validation Loss"])
plt.show()
