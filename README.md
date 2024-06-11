### README.md

# Image Processing and Classification Project

This project consists of various scripts and data for performing image processing tasks such as style transfer, image classification, and optimal size determination. The directory structure and purpose of each script are detailed below.

## Directory Structure

- `contents/` : Contains various content images used for style transfer.
- `multiclass_train/` : Contains training data for the multiclass classifier.
- `styles/` : Contains style images used for style transfer.
- `test_data/` : Contains test data used for evaluating models.
- `train_data/` : Contains training data for image classification models.

## Scripts

- `classifier.py` : This script is used for training and evaluating an image classifier.
- `multiclass_contrastive.py` : This script is used for training a contrastive learning model for multiclass classification.
- `optimal_size.py` : This script determines the optimal size for input images to improve model performance.
- `style_transfer.py` : This script performs style transfer on images using a pre-trained VGG19 model.

## Pre-trained Models

- `vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5` : Pre-trained VGG19 model weights without the top fully connected layers.

## Getting Started

### Prerequisites

Make sure you have Python 3.x installed on your machine. You can install the required packages using the `requirements.txt` file.

### Installation

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

### Usage

1. **Training the Classifier:**

   ```bash
   python classifier.py
   ```

   This script trains an image classifier using the data in the `train_data/` directory. In the `train_data/` directory, there are two folders: `positive` and `negative`. In the `positive` folder, you can place images that correspond to the label you want to predict (e.g., `cat_style1` for cat prediction or `dog_style1` for dog prediction).

2. **Multiclass Contrastive Learning:**

   ```bash
   python multiclass_contrastive.py
   ```

   This script trains a contrastive learning model for multiclass classification using the data in the `multiclass_train/` directory.

3. **Determining Optimal Image Size:**

   ```bash
   python optimal_size.py
   ```

   This script determines the optimal size for input images to improve model performance. The `optimal_size.py` script provides a graph to help find the optimal `k`.

4. **Style Transfer:**

   ```bash
   python style_transfer.py
   ```

   This script performs style transfer on images using the content images in the `contents/` directory and style images in the `styles/` directory. To produce all style-transferred images, download all content and style images from the Google Drive link provided in the report, and place them in the respective `contents/` and `styles/` directories. Then, run the `style_transfer.py` script.

### Data Arrangement

You can arrange the test data in the `test_data/` directory, and the training data in the `train_data/` directory. Style images should be placed in the `styles/` directory, and content images in the `contents/` directory.

### Contributing

If you would like to contribute to this project, please fork the repository and submit a pull request.
