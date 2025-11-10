# Image Classification for Sustainable Fashion
[![Ask DeepWiki](https://devin.ai/assets/askdeepwiki.png)](https://deepwiki.com/Rhea-15/image_classification.git)

This repository contains an AI-powered image classification model to identify the material composition of clothing items from product images. By classifying fabrics as "Natural" or "Synthetic," this project aims to provide consumers with transparent sustainability insights, promoting more responsible purchasing decisions.

## Problem Statement

India’s textile and apparel sector is a significant contributor to the national economy, but it also generates approximately 7.8 million tonnes of waste annually, accounting for 8.5% of global textile waste. With only 34% of this waste being reused and 25% recycled, the environmental impact is substantial.

Simultaneously, consumer demand for sustainability is on the rise. Studies show that 89% of Indian consumers report purchasing eco-friendly fashion, and 80% express strong environmental concern. However, access to clear and consistent sustainability data for fashion products on digital platforms remains limited.

## Our Solution

This project addresses the information gap by providing an image classification system that analyzes clothing images to determine their material type.

The core of the solution is a Convolutional Neural Network (CNN) trained to classify fabric images into two primary categories:
*   **Natural** (e.g., cotton, denim, wool)
*   **Synthetic** (e.g., polyester, nylon, fleece)

This classification helps assess a key sustainability indicator, empowering consumers and promoting brands that use eco-friendly materials.

### Model Architecture

The model is a custom CNN built using TensorFlow and Keras. It is designed for efficiency and accuracy in classifying fabric textures.

The architecture consists of:
1.  **Input Layer**: Accepts images of size (224, 224, 3).
2.  **Data Augmentation**: Applies random horizontal flips, rotations, and zooms to increase dataset variance and prevent overfitting.
3.  **Normalization**: Rescales pixel values from [0, 255] to [0, 1].
4.  **Convolutional Blocks**: Three sequential blocks, each containing:
    *   `Conv2D` layer (32, 64, and 128 filters respectively) with a 'relu' activation.
    *   `BatchNormalization` to stabilize and accelerate training.
    *   `MaxPooling2D` to reduce spatial dimensions.
5.  **Global Average Pooling**: Flattens the feature maps into a single vector per feature map.
6.  **Dense Layers**: A `Dense` hidden layer with 32 units and 'relu' activation, followed by a `Dropout` layer (rate of 0.5) to reduce overfitting.
7.  **Output Layer**: A `Dense` layer with 2 units and a 'softmax' activation to output the probability for the 'Natural' and 'Synthetic' classes.

The model is compiled with the `adam` optimizer and `sparse_categorical_crossentropy` loss function.

### Dataset

The model was trained on the **TextileNet Dataset**, which contains images of various fabric types. The training script (`fashion_based_image_classification.py`) maps 27 distinct fabric categories into the two target classes: 'Natural' and 'Synthetic'.

-   **Natural**: `canvas`, `chambray`, `corduroy`, `crepe`, `denim`, `flannel`, `gingham`, `lace`, `lawn`, `serge`, `tweed`, `twill`
-   **Synthetic**: `chenille`, `chiffon`, `faux_fur`, `faux_leather`, `fleece`, `jersey`, `knit`, `neoprene`, `organza`, `plush`, `satin`, `taffeta`, `tulle`, `velvet`, `vinyl`

## Getting Started

The training script is designed to run in a Google Colab environment.

### Prerequisites
- Python 3.x
- TensorFlow
- A Google Account with Google Drive access

### Setup and Training

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/rhea-15/image_classification.git
    cd image_classification
    ```

2.  **Prepare the Dataset:**
    *   Download the fabric dataset (e.g., from Kaggle's TextileNet).
    *   Create a `fabric.zip` file containing `train` and `test` directories, each with subdirectories for the fabric categories.
    *   Upload `fabric.zip` to a folder in your Google Drive (e.g., `My Drive/image classification/`).

3.  **Run the Training Script:**
    *   Open `fashion_based_image_classification.py` in Google Colab.
    *   The script will prompt you to mount your Google Drive. Authorize the access.
    *   Update the file paths in the script if you placed `fabric.zip` in a different location.
    *   Run all cells in the notebook. The script will:
        1.  Unzip the dataset into the Colab environment.
        2.  Create TensorFlow data pipelines.
        3.  Preprocess the data and map labels to 'Natural'/'Synthetic'.
        4.  Build, compile, and train the model.
        5.  Save the trained model as `material_model.keras` to your specified Google Drive path.

## Sustainability Impact

This solution directly supports the following UN Sustainable Development Goals (SDGs):
*   **SDG 12: Responsible Consumption and Production**: By providing transparency that enables consumers to make informed, sustainable choices.
*   **SDG 13: Climate Action**: By encouraging the use of natural, biodegradable materials over petroleum-based synthetic fabrics, thereby reducing the carbon footprint of the fashion industry.

## File Descriptions

-   `fashion_based_image_classification.py`: The Python script used for data loading, preprocessing, model definition, training, and evaluation in a Google Colab environment.
-   `material_model.keras`: The pre-trained Keras model for classifying fabric images as 'Natural' or 'Synthetic'.