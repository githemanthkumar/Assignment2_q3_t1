import tensorflow as tf
from tensorflow.keras import layers, models

# Define the AlexNet model
def alexnet_model():
    model = models.Sequential()

    # First Conv2D layer: 96 filters, kernel size = (11,11), stride = 4, activation = ReLU
    model.add(layers.Conv2D(96, (11, 11), strides=4, activation='relu', input_shape=(227, 227, 3)))

    # MaxPooling Layer: pool size = (3,3), stride = 2
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=2))

    # Second Conv2D layer: 256 filters, kernel size = (5,5), activation = ReLU
    model.add(layers.Conv2D(256, (5, 5), activation='relu', padding='same'))

    # MaxPooling Layer: pool size = (3,3), stride = 2
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=2))

    # Third Conv2D layer: 384 filters, kernel size = (3,3), activation = ReLU
    model.add(layers.Conv2D(384, (3, 3), activation='relu', padding='same'))

    # Fourth Conv2D layer: 384 filters, kernel size = (3,3), activation = ReLU
    model.add(layers.Conv2D(384, (3, 3), activation='relu', padding='same'))

    # Fifth Conv2D layer: 256 filters, kernel size = (3,3), activation = ReLU
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))

    # MaxPooling Layer: pool size = (3,3), stride = 2
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=2))

    # Flatten Layer
    model.add(layers.Flatten())

    # Fully Connected Layer (Dense): 4096 neurons, activation = ReLU
    model.add(layers.Dense(4096, activation='relu'))

    # Dropout Layer: 50%
    model.add(layers.Dropout(0.5))

    # Fully Connected Layer (Dense): 4096 neurons, activation = ReLU
    model.add(layers.Dense(4096, activation='relu'))

    # Dropout Layer: 50%
    model.add(layers.Dropout(0.5))

    # Output Layer: 10 neurons, activation = Softmax
    model.add(layers.Dense(10, activation='softmax'))

    return model

# Create the model
model = alexnet_model()

# Print the model summary
model.summary()
