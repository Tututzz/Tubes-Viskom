# Tubes-Viskom: CXR Image Classification for COVID-19 Detection

This project, "Tubes-Viskom," is a web application developed to classify chest X-ray (CXR) images. The primary goal is to assist in identifying potential respiratory conditions by predicting whether a CXR image indicates a **COVID-19** infection, a generic **viral** infection, or appears **normal**.

The application utilizes a deep learning model built with TensorFlow and Keras, specifically employing the ConvNeXtTiny architecture as its base.

## Features

* **Image Upload:** Users can upload CXR images in common formats (PNG, JPG, JPEG).
* **Classification:** The uploaded image is processed and fed into the trained model.
* **Prediction Output:** The model predicts one of three classes:
    * `covid`
    * `normal`
    * `virus`
* **Result Display:** The application displays the predicted class along with the model's confidence score for that prediction.
* **Web Interface:** A simple web interface built with Flask allows for easy interaction.

## How It Works

1.  A user uploads a Chest X-Ray image through the web interface.
2.  The image is preprocessed: resized to 224x224 pixels and normalized.
3.  The preprocessed image is passed to the loaded deep learning model (`model_opt-adam_lr-1e-05_bs-32.h5`).
4.  The model, based on ConvNeXtTiny architecture, predicts the likelihood of the image belonging to each of the three classes ('covid', 'normal', 'virus').
5.  The class with the highest probability is presented as the prediction, along with the confidence percentage.

## Technical Stack

* **Backend:** Python, Flask
* **Machine Learning:** TensorFlow, Keras
* **Model Architecture:** ConvNeXtTiny (base)
* **Deployment:** Configured for deployment to Azure Web App via GitHub Actions.
* **Dependencies:** See `requirements.txt` for a full list of Python packages.

## Model Loading
The application is designed to download the model weights (`model_opt-adam_lr-1e-05_bs-32.h5`) from Google Drive if they are not found locally in the `models/` directory. Ensure the `MODEL_FILE_ID` environment variable is correctly set for this functionality.

## Usage

1.  Navigate to the application's URL.
2.  Click the "Choose File" button to select a CXR image.
3.  Click "Upload and Predict" to submit the image for classification.
4.  The results page will display the uploaded image, the predicted category (Covid, Normal, or Virus), and the confidence level of the prediction.
