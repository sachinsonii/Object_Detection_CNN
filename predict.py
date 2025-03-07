import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

# Load the trained classification model
classification_model = tf.keras.models.load_model('object_detection_model.h5')

# Define class labels
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Function to create the object detection model
def create_detection_model(classification_model):
    # Remove the regression output from the classification model
    detection_model = tf.keras.Model(
        inputs=classification_model.input,
        outputs=classification_model.get_layer('classification').output
    )
    return detection_model

# Create the detection model
detection_model = create_detection_model(classification_model)

# Function to perform object detection using the detection model
def detect_objects(image_path, detection_model):
    # Load and preprocess the image
    image = cv2.imread(image_path)
    original_image = image.copy()  # Copy of the original image for drawing
    image = cv2.resize(image, (32, 32)) / 255.0
    image = np.expand_dims(image, axis=0)

    # Perform object detection
    predictions = detection_model.predict(image)

    # Get the predicted class
    predicted_class_idx = np.argmax(predictions)
    predicted_class = labels[predicted_class_idx]
    print(predicted_class)

    # Convert image to grayscale
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray_image, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize the coordinates for the larger bounding box
    min_x, min_y = np.inf, np.inf
    max_x, max_y = -np.inf, -np.inf

    # Find small bounding boxes and update the larger bounding box coordinates
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        

        # Update the larger bounding box coordinates
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x + w)
        max_y = max(max_y, y + h)

    # Draw the larger bounding box around all smaller ones
    if min_x < np.inf and min_y < np.inf and max_x > -np.inf and max_y > -np.inf:
        cv2.rectangle(original_image, (min_x, min_y), (max_x, max_y), (255, 0, 0), 3)  # Red bounding box

    # Add the predicted class label to the image
    cv2.putText(original_image, predicted_class,
                (min_x, min_y - 10),  # Place the label above the larger bounding box
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 0, 0), 2)

    return original_image

# Test the detection model
image_path = 'test6.jfif'  # Replace with your image path
result_image = detect_objects(image_path, detection_model)

# Plot the result image
plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('Object Detection Result')
plt.show()
