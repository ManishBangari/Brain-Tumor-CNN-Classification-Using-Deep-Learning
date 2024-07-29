# Brain-Tumor-CNN-Classification-Using-Deep-Learning

## Introduction

Brain tumors are abnormal growths of cells within the brain, and their early and accurate detection is critical for effective treatment. Traditional diagnostic methods involve manual examination of medical images by radiologists, which can be time-consuming and susceptible to human error. Leveraging the power of deep learning, specifically Convolutional Neural Networks (CNNs), this project aims to automate the classification of brain tumors, enhancing diagnostic accuracy and efficiency.


## Project Overview

This project implements a CNN-based approach to classify brain tumor images into different categories. The model is trained on a dataset of brain MRI images, learning to distinguish between different types of tumors (e.g., glioma, meningioma, pituitary) and healthy brain tissues. The ultimate goal is to develop a reliable and robust system that can assist medical professionals in the early detection and classification of brain tumors.

## Features

- Data Preprocessing: Loading and preprocessing MRI images to ensure they are suitable for training the CNN.
- Model Architecture: Designing and implementing a deep learning model using CNNs tailored for image classification tasks.
- Training and Evaluation: Training the model on the prepared dataset and evaluating its performance using various metrics.
- Visualization: Visualizing the results, including loss curves, accuracy curves, and example predictions.

## Dataset

The dataset used in this project consists of brain MRI images that are publicly available and labeled for different types of brain tumors. The images are preprocessed to standardize their size and enhance their quality for better training results.

## Requirements

- Python 3.10.12
- TensorFlow 2.16.2
- NumPy
- pandas 
- dvc
- mlflow
- Matplotlib
- scipy
- Flask
- seaborn
