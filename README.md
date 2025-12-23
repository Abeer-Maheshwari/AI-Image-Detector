# AI Image Classifier
**Author**: Abeer Maheshwari

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
##
A lightweight **AI-generated image detector** implemented using a small Convolutional Neural Network (CNN). This project builds upon my earlier BeyondAI work and trains a simple binary classifier to distinguish between real images of human faces and AI-generated ones.

This repository includes files for:
- Model training with multiple optimisers
- Loss landscape visualisation
- A simple web application for testing uploaded images

## Features

- Compact CNN architecture for binary classification
- Automatic iteration through multiple optimisers during training
- 1D linear interpolation loss landscape visualisation
- Simple web GUI for uploading and classifying images

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Abeer-Maheshwari/AI-Image-Detector.git
   cd AI-Image-Detector
2. Install the required dependencies:
   ```bash
   pip install torch torchvision torchaudio
   pip install streamlit pillow matplotlib numpy

## Usage
### Training the Model
1. Tweak Hyperparameters to fine tune the model to your liking (optional)
2. Train the model (or alternatively, use my pretrained model https://www.dropbox.com/scl/fi/rvvflwxqjv7xlvntw7ce6/trained_models.pth?rlkey=auc5xwn3hwgu9vyevvgstahvq&st=47tq9xi3&dl=0)
3. Place the resultant file (trained_models.pth) in the same directory as the rest of the files


### Visualise Loss Landscape
Generate and display 1D linearly interpolated loss function graphs by running CNN_inference.py

### Run the Image Classifier
```bash
streamlit run app.py
```
This will open a local web interface where you can upload images and receive a prediction.


## Notes
- Accuracy depends heavily on the training dataset and the specific AI generation methods tested.
- This is an experimental, lightweight project intended for educational purposes or quick prototyping. For production-grade detection, consider more advanced or ensemble-based solutions.
- Contributions, bug reports, and feature requests are welcome! Feel free to open an issue or submit a pull request.

