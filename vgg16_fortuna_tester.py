import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model

saved_model = load_model("vgg16tuna_base.h5")
class_indices = {'bigeye': 0, 'skipjack': 1, 'yellowfin': 2}

def load_and_preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0  # Scale pixel values
    return np.expand_dims(img_array, axis=0)

image_paths = [
    'path/to/image1.jpg',
    'path/to/image2.jpg',
    'path/to/image3.jpg'
]

images = np.vstack([load_and_preprocess_image(img_path) for img_path in image_paths])

predictions = saved_model.predict(images)

predicted_classes = np.argmax(predictions, axis=1)

for img_path, pred in zip(image_paths, predicted_classes):
    print(f'Image: {img_path}, Predicted Class: {class_indices[pred]}')