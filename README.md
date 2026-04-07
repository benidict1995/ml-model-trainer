# ML Model Trainer

A simple and customizable **machine learning model training pipeline** using TensorFlow/Keras.  
This project allows you to train image classification models and convert them into **TensorFlow Lite (TFLite)** format for mobile deployment (e.g., Android).

---

## Features

- Train image classification models from scratch
- Supports custom datasets
- Easy model customization
- Export trained model to `.tflite`
- Clean and modular Python scripts
- Beginner-friendly structure

---

## Project Structure

```
ml-model-trainer/
│
├── train_model.py        # Script to train the model
├── convert_tflite.py     # Convert trained model to TFLite
├── dataset/              # Your dataset (images organized by class)
├── models/               # Saved trained models
├── tflite/               # Exported TFLite models
├── requirements.txt      # Dependencies
└── README.md
```

---

## Requirements

- Python 3.8+
- pip

---

## Installation

```
git clone https://github.com/benidict1995/ml-model-trainer.git
cd ml-model-trainer
```

### Create virtual environment

```
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

---

## Required Libraries

Install all dependencies:

```
pip install -r requirements.txt
```

Or manually install:

```
pip install tensorflow numpy matplotlib pillow scikit-learn
```

---

## Dataset Structure

```
dataset/
│
├── cat/
│   ├── img1.jpg
│   ├── img2.jpg
│
├── dog/
│   ├── img1.jpg
│   ├── img2.jpg
```

---

## Train the Model

```
python train_model.py
```

---

## Convert to TFLite

```
python convert_tflite.py
```

Output:
```
/tflite/model.tflite
```

---

## Common Issues

### TensorFlow install issue
```
pip install tensorflow --upgrade
```

### Out of memory
- Reduce image size
- Reduce batch size

---

## Author

Benidict Dulce
