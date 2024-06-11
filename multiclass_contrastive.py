import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.optimizers import Adam
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from itertools import combinations


def create_pairs(images, labels, file_names):
    """Create pairs of samples to classfy them."""
    pairs = []
    pair_labels = []
    pos_pairs = []
    neg_pairs = []
    num_classes = len(np.unique(labels))
    digit_indices = {label: np.where(labels == label)[0] for label in np.unique(labels)}

    for label, indices in digit_indices.items():
        for i, j in combinations(indices, 2):
            pairs.append([images[i], images[j]])
            pair_labels.append(1)
            pos_pairs.append((file_names[i], file_names[j]))

        neg_label_list = [neg_label for neg_label in digit_indices.keys() if neg_label != label]
        for neg_label in neg_label_list:
            neg_indices = digit_indices[neg_label]
            for i in range(len(indices)):
                neg_img = random.choice(neg_indices)
                pairs.append([images[indices[i]], images[neg_img]])
                pair_labels.append(0)
                neg_pairs.append((file_names[indices[i]], file_names[neg_img]))
    
    return np.array(pairs), np.array(pair_labels), pos_pairs, neg_pairs


def load_images_and_labels(directory):
    """Load images and their labels."""
    images = []
    labels = []
    file_names = []
    for file in os.listdir(directory):
        if file.endswith('.png'):
            label, style = file.split('_')[:2] # cat_da-vinci: get "cat"
            img_path = os.path.join(directory, file)
            img = load_img(img_path, target_size=(150, 150))
            img = img_to_array(img) / 255.0
            images.append(img)
            labels.append(label)
            file_names.append(file)
    return np.array(images), np.array(labels), file_names


train_data_dir = 'multiclass_train'
test_data_dir = 'test_data'


train_images, train_labels, train_file_names = load_images_and_labels(train_data_dir)
test_images, test_labels, test_file_names = load_images_and_labels(test_data_dir)


train_pairs, train_pair_labels, train_pos_pairs, train_neg_pairs = create_pairs(train_images, train_labels, train_file_names)
test_pairs, test_pair_labels, test_pos_pairs, test_neg_pairs = create_pairs(test_images, test_labels, test_file_names)


print(f"Total number of training pairs: {len(train_pairs)}")
print(f"Number of positive training pairs: {len(train_pos_pairs)}")
print(f"Number of negative training pairs: {len(train_neg_pairs)}")
print(f"Total number of testing pairs: {len(test_pairs)}")
print(f"Number of positive testing pairs: {len(test_pos_pairs)}")
print(f"Number of negative testing pairs: {len(test_neg_pairs)}")


print("\nPositive training pairs (file names):")
for pair in train_pos_pairs[:5]:  # Print first 5 positive pairs
    print(pair)


print("\nNegative training pairs (file names):")
for pair in train_neg_pairs[:5]:  # Print first 5 negative pairs
    print(pair)


def contrastive_loss(y_true, y_pred, margin=1):
    y_true = tf.cast(y_true, y_pred.dtype)
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

def build_siamese_model(input_shape):
    base_model = tf.keras.Sequential([
        layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(512, activation='relu')
    ])

    input_a = layers.Input(shape=input_shape)
    input_b = layers.Input(shape=input_shape)

    processed_a = base_model(input_a)
    processed_b = base_model(input_b)

    distance = layers.Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))([processed_a, processed_b])
    output = layers.Dense(1, activation='sigmoid')(distance)

    model = tf.keras.Model([input_a, input_b], output)
    return model

input_shape = (150, 150, 3)
siamese_model = build_siamese_model(input_shape)
siamese_model.compile(loss=contrastive_loss, optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])
siamese_model.summary()

history = siamese_model.fit(
    [train_pairs[:, 0], train_pairs[:, 1]], train_pair_labels,
    batch_size=20,
    epochs=30,
    validation_data=([test_pairs[:, 0], test_pairs[:, 1]], test_pair_labels)
)

def classify_image_pair(img_path1, img_path2):
    img1 = load_img(img_path1, target_size=(150, 150))
    img1 = img_to_array(img1) / 255.0
    img1 = np.expand_dims(img1, axis=0)

    img2 = load_img(img_path2, target_size=(150, 150))
    img2 = img_to_array(img2) / 255.0
    img2 = np.expand_dims(img2, axis=0)

    prediction = siamese_model.predict([img1, img2])
    print(f"Prediction: {prediction[0][0]}")
    return 'negative' if prediction[0] < 0.5 else 'positive'

def visualize_predictions(image_pairs):
    for img_path1, img_path2 in image_pairs:
        img1 = load_img(img_path1, target_size=(150, 150))
        img2 = load_img(img_path2, target_size=(150, 150))

        prediction = classify_image_pair(img_path1, img_path2)
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(img1)
        plt.title('Image 1')

        plt.subplot(1, 2, 2)
        plt.imshow(img2)
        plt.title('Image 2')

        plt.suptitle(f"Predicted: {prediction}")
        plt.show()

def get_image_pairs_from_directory(directory_path):
    image_pairs = []
    image_files = [file for file in os.listdir(directory_path) if file.endswith('.png')]
    for file_name1, file_name2 in combinations(image_files, 2):
        image_pairs.append((os.path.join(directory_path, file_name1), os.path.join(directory_path, file_name2)))
    return image_pairs

test_data_directory = "test_data"
test_image_pairs = get_image_pairs_from_directory(test_data_directory)
visualize_predictions(test_image_pairs)


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


## The code worked for simplier data but not worked for whole.
# It can be furter improved.