import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image
import numpy as np
import os


base_dir = 'train_data'
train_dir = os.path.join(base_dir)


train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # Use 20% of the data for validation
)

# Load images from directories
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(500, 500),
    batch_size=20,
    class_mode='binary',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(500, 500),
    batch_size=20,
    class_mode='binary',
    subset='validation'
)

# Build a CNN
model = models.Sequential([
    layers.Conv2D(16, (3, 3), activation='relu', input_shape=(500, 500, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch= train_generator.samples // train_generator.batch_size,
    epochs=30,
    validation_data = validation_generator,
    validation_steps = validation_generator.samples // validation_generator.batch_size
)

def classify_image(img_path):
    """Predict test images"""
    img = image.load_img(img_path, target_size=(500, 500))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.

    prediction = model.predict(img_tensor)
    print(f"Prediction: {prediction[0][0]}")
    return 'negative' if prediction[0] < 0.5 else 'positive'

print("Class indices:", train_generator.class_indices)
print("Number of training samples:", train_generator.samples)
print("Number of validation samples:", validation_generator.samples)

import matplotlib.pyplot as plt

def visualize_predictions(img_paths):
    """Visualize the predictions of test images"""
    for img_path in img_paths:
        img = image.load_img(img_path, target_size=(500, 500))
        img_tensor = image.img_to_array(img)
        img_tensor = np.expand_dims(img_tensor, axis=0)
        img_tensor /= 255.

        prediction = model.predict(img_tensor)
        label = 'negative' if prediction[0] < 0.5 else 'positive'

        plt.imshow(img)
        plt.title(f"Predicted: {label}")
        plt.show()


def get_image_names_from_directory(directory_path):
    """Get test images from the test directory."""
    image_files = []
    for file_name in os.listdir(directory_path):
        if file_name.endswith('.png'):  
            image_files.append(os.path.join(directory_path, file_name))
    return image_files

test_data_directory = "test_data"
test_images = get_image_names_from_directory(test_data_directory)

print(test_images)

visualize_predictions(test_images)


# Finally plot accuracy and loss graphs
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()