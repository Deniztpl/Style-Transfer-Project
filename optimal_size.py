import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt
import os
import time

# Directories for style and content images
stylesDir = "styles/"
contentsDir = "contents/"
# Different sizes to test
testSizes = [100, 300, 500, 750, 1000]

# Selected content and style images
contentImageName = "cat.jpg"
styleImageName = "edvard-munch.jpg"

contentImagePath = os.path.join(contentsDir, contentImageName)
styleImagePath = os.path.join(stylesDir, styleImageName)

# Function to preprocess the image
def preprocessImage(imagePath, targetSize):
    """
    Load and preprocess the image for VGG19.
    Resizes the image to maintain the aspect ratio for the given target size.
    """
    img = load_img(imagePath)
    originalSize = img.size
    aspectRatio = originalSize[0] / originalSize[1]

    if aspectRatio > 1:  # Width is greater than height
        newWidth = targetSize
        newHeight = int(newWidth / aspectRatio)
    else:  # Height is greater than or equal to width
        newHeight = targetSize
        newWidth = int(newHeight * aspectRatio)

    img = img.resize((newWidth, newHeight))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.vgg19.preprocess_input(img)

    return img

# Function to deprocess the image
def deprocessImage(x):
    """
    Convert a processed image tensor back to a viewable image.
    This function reverses the preprocessing steps applied.
    """
    x = x.reshape((x.shape[1], x.shape[2], x.shape[3]))
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]  # Convert from BGR to RGB
    x = np.clip(x, 0, 255).astype('uint8')
    return x

# Function to get the VGG19 model with selected layers
def getVGGModel(styleLayers, contentLayers):
    """
    Load VGG19 model and extract the outputs for given style and content layers.
    """
    vgg = VGG19(weights="vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5", include_top=False)
    outputs = [vgg.get_layer(name).output for name in styleLayers + contentLayers]
    model = Model([vgg.input], outputs)
    return model

# Function to compute Gram matrix
def gramMatrix(inputTensor):
    """
    Compute the Gram matrix for an input tensor. Used to calculate style loss.
    """
    result = tf.linalg.einsum('bijc,bijd->bcd', inputTensor, inputTensor)
    inputShape = tf.shape(inputTensor)
    numLocations = tf.cast(inputShape[1] * inputShape[2], tf.float32)
    return result / numLocations

# Lists to store results
timeResults = []
lossResults = []

# Processing for each target size
for testSize in testSizes:
    startTime = time.time()

    # Load and preprocess content and style images
    contentImg = preprocessImage(contentImagePath, testSize)
    styleImg = preprocessImage(styleImagePath, testSize)

    # Define style and content layers
    styleLayers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
    contentLayers = ['block5_conv2']

    # Get VGG19 model with specified layers
    model = getVGGModel(styleLayers, contentLayers)

    # Extract features for style and content images
    styleOutputs = model(styleImg)
    contentOutputs = model(contentImg)

    # Compute Gram matrices for each style layer's output
    gramStyleFeatures = [gramMatrix(feature) for feature in styleOutputs[:len(styleLayers)]]

    # Content features to be used for optimization
    contentFeatures = [contentFeature for contentFeature in contentOutputs[len(styleLayers):]]

    # Initialize image for optimization
    initImg = tf.Variable(contentImg, dtype=tf.float32)

    # Style and content weights
    styleWeight = 1.0
    contentWeight = 0.025

    # Optimizer setup
    optimizer = tf.optimizers.Adam(learning_rate=4.0, beta_1=0.99, epsilon=1e-1)

    # Training step function
    @tf.function
    def trainStep(image):
        """
        Performs a single step of the optimization to update the image.
        Computes loss and applies gradients.
        """
        with tf.GradientTape() as tape:
            outputs = model(image)
            styleOutputs, contentOutputs = outputs[:len(styleLayers)], outputs[len(styleLayers):]
            styleScore = tf.add_n([tf.reduce_mean((gramMatrix(style) - target) ** 2)
                                    for style, target in zip(styleOutputs, gramStyleFeatures)])
            contentScore = tf.add_n([tf.reduce_mean((content - target) ** 2)
                                      for content, target in zip(contentOutputs, contentFeatures)])
            loss = styleWeight * styleScore + contentWeight * contentScore

        grad = tape.gradient(loss, image)
        optimizer.apply_gradients([(grad, image)])
        image.assign(tf.clip_by_value(image, -103.939, 151.061))

        return loss

    # Run the optimization for a fixed number of iterations
    losses = []
    for i in range(1, 1001):
        loss = trainStep(initImg)
        losses.append(loss.numpy())
        if i % 100 == 0:
            print(f"Iteration {i} completed for testSize {testSize}")

    # Calculate time spent and average loss
    endTime = time.time()
    totalTime = endTime - startTime
    avgLoss = np.mean(losses)

    # Store results
    timeResults.append(totalTime)
    lossResults.append(avgLoss)

# Plot results
plt.figure(figsize=(12, 6))

# Plot time spent
plt.subplot(1, 2, 1)
plt.plot(testSizes, timeResults, marker='o', linestyle='-')
plt.xlabel('Image Size (testSize)')
plt.ylabel('Time Spent (seconds)')
plt.title('Time Spent vs. Image Size')

# Plot average loss
plt.subplot(1, 2, 2)
plt.plot(testSizes, lossResults, marker='o', linestyle='-')
plt.xlabel('Image Size (testSize)')
plt.ylabel('Average Loss')
plt.title('Average Loss vs. Image Size')

plt.tight_layout()
plt.show()
