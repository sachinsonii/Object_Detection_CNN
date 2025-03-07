import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ReduceLROnPlateau

# Load CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Normalize and preprocess data
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Label names
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# One-hot encode the labels
encoder = OneHotEncoder(categories='auto', sparse_output=False)
y_train_onehot = encoder.fit_transform(y_train)
y_test_onehot = encoder.transform(y_test)

# Split the dataset into training and validation sets
X_train, X_val, y_train_onehot, y_val_onehot = train_test_split(
    X_train, y_train_onehot, test_size=0.2, random_state=42
)

# Data augmentation setup
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
datagen.fit(X_train)

# Define the CNN model for object detection
def create_cnn_model(input_shape, num_classes):
    input_layer = Input(shape=input_shape)

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    x = Flatten()(x)

    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)  # Dropout layer to prevent overfitting
    x = BatchNormalization()(x)
    
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)  # Another dropout layer
    x = BatchNormalization()(x)

    classification_output = Dense(num_classes, activation='softmax', name='classification')(x)
    regression_output = Dense(4, name='regression')(x)  # 4 for (x, y, w, h) of bounding box

    model = tf.keras.Model(inputs=input_layer, outputs=[classification_output, regression_output])

    return model

# Define the input shape and number of classes
input_shape = (32, 32, 3)
num_classes = 10

# Create the model
with tf.device('/GPU:0'):  # Ensure model uses the GPU if available
    model = create_cnn_model(input_shape, num_classes)

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
losses = {
    'classification': 'categorical_crossentropy',
    'regression': 'mean_squared_error'
}
metrics = {
    'classification': 'accuracy',
    'regression': 'mse'
}

model.compile(optimizer=optimizer, loss=losses, metrics=metrics)

# Learning rate reduction callback
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

# Train the model
history = model.fit(
    datagen.flow(X_train, {'classification': y_train_onehot, 'regression': np.zeros((y_train_onehot.shape[0], 4))}),
    epochs=50,
    validation_data=(X_val, {'classification': y_val_onehot, 'regression': np.zeros((y_val_onehot.shape[0], 4))}),
    steps_per_epoch=len(X_train) // 32,
    callbacks=[reduce_lr]
)

# Evaluate the model
results = model.evaluate(
    X_test,
    {'classification': y_test_onehot, 'regression': np.zeros((y_test_onehot.shape[0], 4))}
)

# Extract the values from the results
total_loss, classification_loss, classification_accuracy = results[:3]

# Print the results
print(f"Classification Loss: {classification_loss}")
print(f"Classification Accuracy: {classification_accuracy}")

# Save the trained model
model.save('object_detection_model.h5')
print("Model saved successfully!")
