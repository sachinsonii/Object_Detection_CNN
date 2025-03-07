# Object Detection with CNN

This project implements an object detection model using Convolutional Neural Networks (CNNs) trained on the CIFAR-10 dataset. The model is capable of classifying images into 10 categories and detecting objects within an image using contour detection.

## Features

- **CNN-based Classification**: Uses TensorFlow and Keras to classify images into 10 categories.
- **Bounding Box Detection**: Uses OpenCV to detect object boundaries.
- **Data Augmentation**: Implements data augmentation techniques for better generalization.
- **GPU Support**: The model automatically utilizes the GPU if available.
- **Pretrained Model Usage**: Supports loading and using a saved model for inference.

## Dataset

The model is trained on the CIFAR-10 dataset, which contains 60,000 32x32 color images across 10 categories:

```
['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
```

## Installation

### Prerequisites

Ensure you have the required dependencies installed:

```sh
pip install numpy tensorflow matplotlib opencv-python scikit-learn
```

### Clone the Repository

```sh
git clone https://github.com/sachinsonii/Object_Detection_CNN.git
cd Object_Detection_CNN
```

## Training the Model

To train the CNN model from scratch, run:

```sh
python main.py
```

This will:

1. Load and preprocess the CIFAR-10 dataset.
2. Train a CNN with classification and bounding box regression outputs.
3. Save the trained model as `object_detection_model.h5`.

## Object Detection

To test object detection on a custom image, run:

```sh
python predict.py
```

Make sure to replace the image path in `predict.py` with your desired test image.

## Model Architecture

The model consists of:

- **Convolutional Layers**: Extract features from images.
- **Batch Normalization**: Normalizes activations to improve stability.
- **MaxPooling Layers**: Reduces spatial dimensions.
- **Fully Connected Layers**: Classifies the object and predicts bounding box coordinates.
- **Output Layers**:
  - `classification` (Softmax activation) - Predicts the object category.
  - `regression` (Linear activation) - Predicts bounding box coordinates (x, y, width, height).

## Results & Performance

After training, the model achieves:

- **Classification Accuracy**: \~80% on CIFAR-10
- **Bounding Box Detection**: Uses edge detection and contour approximation for object localization.

## Future Improvements

- Improve bounding box prediction using CNN-based regression.
- Train on a larger dataset with higher-resolution images.
- Implement YOLO-style detection for real-time performance.

## License

This project is licensed under the MIT License.

