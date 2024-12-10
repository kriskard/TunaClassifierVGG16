import tensorflow as tf
from tensorflow.keras.applications import VGG16 
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, MaxPooling2D, BatchNormalization, Activation, Add, Flatten, Dense, ReLU
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import classification_report, confusion_matrix

# Data generators
train_directory="C:/Users/KRISTIAN/Documents/GitHub/TunaClassifierVGG16/dataset/train"
valid_directory="C:/Users/KRISTIAN/Documents/GitHub/TunaClassifierVGG16/dataset/valid"

trdatagen = ImageDataGenerator(
    rescale=1./255,               
    shear_range=0.2,             
    zoom_range=0.2,               
    horizontal_flip=True          
)
traindata = trdatagen.flow_from_directory(directory=train_directory, target_size=(224, 224), batch_size=32, class_mode='categorical')

valdatagen = ImageDataGenerator(rescale=1./255)
validdata = valdatagen.flow_from_directory(directory=valid_directory, target_size=(224, 224), batch_size=8, class_mode='categorical')

# Model setup
# Function for a convolutional block with ReLU
def conv_block(x, filters, kernel_size=(3, 3), padding="same", activation="relu"):
    x = Conv2D(filters, kernel_size, padding=padding)(x)
    x = Activation(activation)(x)
    return x

# Function for a depthwise separable convolution block
def dsc_block(x, filters, kernel_size=(3, 3), padding="same", activation="relu"):
    x_skip = x  # Saving the original input for residual connection
    
    x = DepthwiseConv2D(kernel_size=kernel_size, padding=padding)(x)
    x = Activation(activation)(x)
    
    # Pointwise convolution (1x1)
    x = Conv2D(filters, (1, 1), padding=padding)(x)
    x = Activation(activation)(x)
    
    # Adding the residual connection (skip connection)
    x = Add()([x, x_skip])  # Skip connection is added to the output
    return x

# Input Layer
inputs = Input(shape=(224, 224, 3))

# Block 1 (64 filters)
x = conv_block(inputs, 64)
x = conv_block(x, 64)
x = dsc_block(x, 64)
x = MaxPooling2D((2, 2))(x)

# Block 2 (128 filters)
x = conv_block(x, 128)
x = conv_block(x, 128)
x = dsc_block(x, 128)
x = MaxPooling2D((2, 2))(x)

# Block 3 (256 filters)
x = conv_block(x, 256)
x = conv_block(x, 256)
x = conv_block(x, 256)
x = dsc_block(x, 256)
x = MaxPooling2D((2, 2))(x)

# Block 4 (512 filters with Depthwise Separable Convolutions)
x = conv_block(x, 512)
x = conv_block(x, 512)
x = conv_block(x, 512)
x = MaxPooling2D((2, 2))(x)

# Block 5 (512 filters with Asymmetric Convolutions)
a1 = Conv2D(512, (3, 1), padding="same", activation="relu")(x)
a1 = Conv2D(512, (1, 3), padding="same", activation="relu")(a1)
a1 = BatchNormalization()(a1)
x = Add()([a1])  # Adding asymmetric convolution

a2 = Conv2D(512, (3, 1), padding="same", activation="relu")(x)
a2 = Conv2D(512, (1, 3), padding="same", activation="relu")(a2)
a2 = BatchNormalization()(a2)
x = Add()([a2]) 

a3 = Conv2D(512, (3, 1), padding="same", activation="relu")(x)
a3 = Conv2D(512, (1, 3), padding="same", activation="relu")(a3)
a3 = BatchNormalization()(a3)
x = Add()([a3]) 

x = MaxPooling2D((2, 2))(x)

# Fully Connected Layers
x = Flatten()(x)
x = Dense(4096, activation='relu')(x)
x = Dense(4096, activation='relu')(x)
outputs = Dense(3, activation='softmax')(x)

# Model Creation
model = Model(inputs, outputs)

# Amount of data and summary
print(f"Total training images: {traindata.samples}")
print(f"Total validation images: {validdata.samples}")
model.summary()

# Compile model
opt = Adam(learning_rate=0.0001)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
checkpoint = ModelCheckpoint("vgg16tuna_mlr.keras", monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
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

saved_model = load_model("vgg16tuna_mlr.keras")
predictions = saved_model.predict(validdata)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = validdata.classes
class_labels = list(validdata.class_indices.keys())
report = classification_report(true_classes, predicted_classes, target_names=class_labels)
print(report)   

cm=confusion_matrix(true_classes,predicted_classes)
print(cm)

