import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt
import os
import random

# Directories for saving images
saveDirPositive = "train_data/positive/"
saveDirNegative = "train_data/negative/"
saveDirTest = "test_data"

# Directories for style and content images
styleDir = "styles/"
contentDir = "contents/"

# Lists to store filenames
contentList = [f for f in os.listdir(contentDir) if os.path.isfile(os.path.join(contentDir, f))]
styleList = [f for f in os.listdir(styleDir) if os.path.isfile(os.path.join(styleDir, f))]

print("Style images:", styleList)
print("Content images:", contentList)

def preprocessImage(imagePath, targetSize=750):
    """
    Preprocess the image for VGG19.
    Resize the image while maintaining the aspect ratio.
    """
    img = load_img(imagePath)
    aspectRatio = img.size[0] / img.size[1]

    if aspectRatio > 1:
        newWidth = targetSize
        newHeight = int(newWidth / aspectRatio)
    else:
        newHeight = targetSize
        newWidth = int(newHeight * aspectRatio)

    img = img.resize((newWidth, newHeight))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.vgg19.preprocess_input(img)

    return img

def deprocessImage(imgTensor):
    """
    Convert processed image tensor back to a displayable image.
    """
    imgTensor = imgTensor.reshape((imgTensor.shape[1], imgTensor.shape[2], imgTensor.shape[3]))
    imgTensor[:, :, 0] += 103.939
    imgTensor[:, :, 1] += 116.779
    imgTensor[:, :, 2] += 123.68
    imgTensor = imgTensor[:, :, ::-1]  # BGR to RGB
    imgTensor = np.clip(imgTensor, 0, 255).astype('uint8')
    return imgTensor

def getModel(styleLayers, contentLayers):
    """
    Load VGG19 and extract specific layers' outputs.
    """
    vgg = VGG19(weights="vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5", include_top=False)
    outputs = [vgg.get_layer(name).output for name in styleLayers + contentLayers]
    return Model([vgg.input], outputs)

def computeGramMatrix(tensor):
    """
    Compute the Gram matrix used to calculate style loss.
    """
    result = tf.linalg.einsum('bijc,bijd->bcd', tensor, tensor)
    numLocations = tf.cast(tf.shape(tensor)[1] * tf.shape(tensor)[2], tf.float32)
    return result / numLocations

# Process each content and style image
for contentFile in contentList:
    contentPath = os.path.join(contentDir, contentFile)
    for styleFile in styleList:
        stylePath = os.path.join(styleDir, styleFile)

        # Load and preprocess images
        contentImage = preprocessImage(contentPath)
        styleImage = preprocessImage(stylePath)

        # Define style and content layers
        styleLayers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
        contentLayer = ['block5_conv2']

        # Get model
        model = getModel(styleLayers, contentLayer)

        # Extract style and content features
        styleOutputs = model(styleImage)
        contentOutputs = model(contentImage)

        # Compute Gram matrices for style features
        gramStyleFeatures = [computeGramMatrix(output) for output in styleOutputs[:len(styleLayers)]]

        # Content features for optimization
        contentFeatures = contentOutputs[len(styleLayers):]

        # Initial image for optimization
        initImage = tf.Variable(contentImage, dtype=tf.float32)

        # Weights for style and content
        styleWeight = 1.0
        contentWeight = 0.025

        # Optimizer
        optimizer = tf.optimizers.Adam(learning_rate=4.0, beta_1=0.99, epsilon=1e-1)

        # Training step function
        @tf.function
        def trainStep(image):
            """
            Perform one step of the optimization.
            Compute loss and apply gradients.
            """
            with tf.GradientTape() as tape:
                outputs = model(image)
                styleOutputs, contentOutputs = outputs[:len(styleLayers)], outputs[len(styleLayers):]
                styleLoss = tf.add_n([tf.reduce_mean((computeGramMatrix(style) - target) ** 2)
                                      for style, target in zip(styleOutputs, gramStyleFeatures)])
                contentLoss = tf.reduce_mean((contentOutputs[0] - contentFeatures[0]) ** 2)
                loss = styleWeight * styleLoss + contentWeight * contentLoss

            gradients = tape.gradient(loss, image)
            optimizer.apply_gradients([(gradients, image)])
            image.assign(tf.clip_by_value(image, -103.939, 151.061))

        # Run optimization for a fixed number of iterations
        for iteration in range(1, 1001):
            trainStep(initImage)
            if iteration % 100 == 0:
                print(f"Iteration {iteration} completed for {contentFile} and {styleFile}")

        # Deprocess and display the final image
        finalImage = deprocessImage(initImage.numpy())
        plt.imshow(finalImage)
        plt.show()

        # Save the final image
        contentBase = os.path.splitext(contentFile)[0]
        styleBase = os.path.splitext(styleFile)[0]
        filename = f"{contentBase}_{styleBase}.png"

        # Determine save directory
        # You can change positive sample category to any other category
        # For example, if you want to train a model to detect dogs, you can change "cat" to "dog"
        if "cat" in filename:
            filePath = os.path.join(saveDirPositive, filename)
        else:
            filePath = os.path.join(saveDirNegative, filename)

        # Randomly assign some images to the test set
        if random.random() < 0.2:
            filePath = os.path.join(saveDirTest, filename)
        
        # Save the image
        plt.imsave(filePath, finalImage)
