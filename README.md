# BIO Intelligent Algorithms BIU 2021 Final Projects

This course teaches the foundation of deep neural networks.

In this course we have implemented from scratch, without using any external machine learning related
library 2 types of neural networks.

* Project 1 - A fully connected neural network for performing classification
* Project 2 - A convolutional neural network for performing classification

We have implemented a generic Multi Class Neural Network Python application,
which gives the user the ability to run the model and control its various features for different kind of datasets from the CLI.

We have the support for 2 accelerated implementations:
* GPU accelerated using vectorized implementation with CuPy
* CPU accelerated using vectorized implementation with NumPy

## Usage and installation:
The instructions for installation and usage can be found for each project in the file named: instructions_highlights_and_more.docx

## Repository structure:
    .
    ├── README.md                                       # This README file
    ├── project_1_dnn                                   # Project 1 folder - Fully Connected Deep Neural Network
    │   ├── accelerated_model.py                        # Python implementation of the model
    │   ├── data.zip                                    # The dataset
    │   ├── instructions_highlights_and_more.docx       # Usage instructions
    │   ├── link_to_models_location.txt                 # Google Drive Link to Pickle trained models
    │   ├── models_dump                                 # Pickle dumps of the trained models
    │   │   ├── cupy_trained_model_dump.bin             # GPU accelerated model dump
    │   │   └── numpy_trained_model_dump.bin            # CPU accelerated model dump
    │   ├── output.txt                                  # Test file output predictions
    │   ├── project_instructions.pdf                    # Project instructions
    │   ├── report.docx                                 # Research report
    │   └── whiteboard.svg                              # Whiteboard with model architecture and math
    └── project_2_cnn                                   # Project 2 folder - Convolutional Neural Network
        ├── accelerated_model.py                        # Python implementation of the model
        ├── data.zip                                    # The dataset
        ├── instructions_highlights_and_more.docx       # Usage instructions
        ├── link_to_models_location.txt                 # Google Drive Link to Pickle trained models
        ├── models_dump                                 # Pickle dumps of the trained models
        │   ├── trained_model.bin_cupy                  # GPU accelerated model dump
        │   └── trained_model.bin_numpy                 # CPU accelerated model dump
        ├── output.txt                                  # Test file output predictions
        ├── project_instructions.pdf                    # Project instructions
        ├── report.docx                                 # Research report
        └── whiteboard.svg                              # Whiteboard with model architecture and math

## Colab Notebooks:
The code available in this repository can also be seen in the following Colab Notebooks.

* Project 1 - A fully connected neural network for performing classification:
https://colab.research.google.com/drive/1TW4jpGPCct0mP7RJCzeL25LYe-mEpToi#scrollTo=D8R_QvR-5Mk3

* Project 2 - A convolutional neural network for performing classification:
https://colab.research.google.com/drive/1ky_qjRwA-b4W_exbYlDfsofGHsVSfZy4#scrollTo=D8R_QvR-5Mk3
