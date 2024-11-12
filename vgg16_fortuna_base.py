import tensorflow as tf
from tensorflow.keras.applications import VGG16 
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import classification_report, confusion_matrix

# Data generators
trdatagen = ImageDataGenerator(
    rescale=1./255,               
    shear_range=0.2,             
    zoom_range=0.2,               
    horizontal_flip=True          
)
traindata = trdatagen.flow_from_directory(directory="C:/Users/KRISTIAN/Documents/GitHub/TunaClassifierVGG16/dataset/train", target_size=(224, 224), batch_size=32, class_mode='categorical')

valdatagen = ImageDataGenerator(rescale=1./255)
validdata = valdatagen.flow_from_directory(directory="C:/Users/KRISTIAN/Documents/GitHub/TunaClassifierVGG16/dataset/valid", target_size=(224, 224), batch_size=8, class_mode='categorical')

# Model setup
model = Sequential() 
model.add(Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Flatten()) 
model.add(Dense(512, activation='relu'))
model.add(Dense(3, activation='softmax'))

# Compile model
opt = Adam(learning_rate=0.0001)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# Amount of data and summary
print(f"Total training images: {traindata.samples}")
print(f"Total validation images: {validdata.samples}")
model.summary()

# Callbacks
checkpoint = ModelCheckpoint("vgg16tuna_base.keras", monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto')
early = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=1, mode='auto')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-5, verbose=1)

# Fit model
hist = model.fit(traindata, validation_data=validdata, epochs=100, callbacks=[checkpoint, early, reduce_lr])

print('\a')
# Plotting
plt.plot(hist.history["accuracy"])
plt.plot(hist.history['val_accuracy'])
plt.plot(hist.history['val_loss'])
plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Accuracy", "Validation Accuracy", "Validation Loss"])
plt.show()

saved_model = load_model("vgg16tuna_base.keras")
predictions = saved_model.predict(validdata)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = validdata.classes
class_labels = list(validdata.class_indices.keys())
report = classification_report(true_classes, predicted_classes, target_names=class_labels)
print(report)   

cm=confusion_matrix(true_classes,predicted_classes)
print(cm)

