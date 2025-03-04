# Assignment2_q3_t1
Libraries:

TensorFlow/Keras libraries (tensorflow, layers, and models) are imported for building the neural network.
Defining the AlexNet Model:

A function alexnet_model() is created to build a simplified version of the AlexNet architecture using Keras’ Sequential API.
Layers:

Conv2D Layers: There are five convolutional layers with different configurations:
First Conv layer with 96 filters, an 11x11 kernel, and stride of 4.
Second Conv layer with 256 filters and a 5x5 kernel.
Third and Fourth Conv layers, each with 384 filters and 3x3 kernels.
Fifth Conv layer with 256 filters and a 3x3 kernel.
MaxPooling Layers: Three max-pooling layers reduce the spatial dimensions after some convolution layers using a 3x3 pool size and stride of 2.
Flatten Layer: Converts the 3D feature maps into a 1D vector to feed into fully connected (dense) layers.
Dense Layers: Two fully connected layers, each with 4096 neurons, followed by ReLU activations.
Dropout Layers: Two dropout layers, each with a 50% dropout rate to prevent overfitting.
Output Layer: A final Dense layer with 10 neurons and Softmax activation for multi-class classification.
Model Summary:

After defining the model, the summary() method prints the architecture, showing each layer's output shape and number of parameters.
This code builds a deep neural network inspired by AlexNet for image classification, typically used on datasets like ImageNet or CIFAR-10.
