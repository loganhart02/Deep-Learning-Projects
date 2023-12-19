# LeNet-5 Implementation Notes

## Overview
LeNet-5 is a convolutional neural network (CNN) designed for handwriting recognition and image classification tasks. Originally tailored for 32x32 pixel grayscale images, the architecture consists of alternating convolutional and subsampling (pooling) layers, followed by fully connected layers.

## Architecture
- **Input Layer**: Accepts 32x32 pixel grayscale images.
- **C1 (Convolutional Layer)**: 6 feature maps, 5x5 kernel size, stride 1, no padding.
- **S2 (Subsampling/Pooling Layer)**: 2x2 kernel size, stride 2. Typically, average pooling is used.
- **C3 (Convolutional Layer)**: 16 feature maps, 5x5 kernel size, stride 1, no padding.
- **S4 (Subsampling/Pooling Layer)**: Similar to S2.
- **C5 (Convolutional Layer)**: 120 feature maps, 5x5 kernel size, leading to fully connected layers.
- **F6 (Fully Connected Layer)**: 84 units.
- **Output Layer**: Number of units corresponds to the number of classes (e.g., 10 for digit recognition).

## Modifications for Larger Images
If adapting LeNet-5 for larger images (e.g., 224x224 pixels), significant modifications are needed, especially in the fully connected layers to accommodate the increased tensor size.

### Adjustments
1. **Output Size Calculation**: After each convolutional and pooling layer, recalculate the output size.
2. **Fully Connected Layer Size**: Adjust the input size of the first fully connected layer based on the output tensor's dimensions from the last pooling layer.

## Implementation Considerations
- **Activation Functions**: Originally, tanh or sigmoid activations were used. Modern implementations often use ReLU for better performance.
- **Pooling**: Average pooling can be replaced with max pooling for potentially better feature extraction.
- **Parameter Count**: Adapting the network for larger images can significantly increase the number of parameters, especially in the fully connected layers.


## Calculating Output Size in Convolutional Neural Networks

Understanding how to calculate the output size of layers in a CNN is crucial for designing and adapting network architectures.

### Output Size for Convolutional Layers

The output size (height and width) of a convolutional layer can be calculated using the formula:

\[ \text{Output Size} = \frac{\text{Input Size} - \text{Kernel Size} + 2 \times \text{Padding}}{\text{Stride}} + 1 \]

- **Input Size**: The height or width of the input.
- **Kernel Size**: The size of the filter/kernel used in the convolutional layer.
- **Padding**: The number of pixels added around the border of the input.
- **Stride**: The step size with which the filter moves across the input.

### Output Size for Pooling Layers

For pooling layers like max pooling or average pooling:

\[ \text{Output Size} = \frac{\text{Input Size}}{\text{Pool Size}} \]

assuming the stride is equal to the pool size and there's no padding.

### Example Calculation for LeNet-5

Let's calculate the output size of each layer for a modified LeNet-5 architecture, assuming an input image size of 224x224 pixels.

- **Input Layer**: 224x224 pixels.
- **C1**: Using a 5x5 kernel with stride 1 and no padding, the output size is \( \frac{224 - 5}{1} + 1 = 220 \). Thus, the output is 220x220x6.
- **S2**: With a 2x2 pooling, the output size is \( \frac{220}{2} = 110 \). The output is 110x110x6.
- **C3**: Again using a 5x5 kernel with stride 1 and no padding, the output size is \( \frac{110 - 5}{1} + 1 = 106 \). Thus, the output is 106x106x16.
- **S4**: With a 2x2 pooling, the output size is \( \frac{106}{2} = 53 \). The output is 53x53x16.
- **C5**: Finally, using a 5x5 kernel with stride 1 and no padding, the output size is \( \frac{53 - 5}{1} + 1 = 49 \). The output is 49x49x120.

### Fully Connected Layer Input Size

The size of the input to the first fully connected layer is the product of the dimensions of the output tensor from the last pooling or convolutional layer. For our example, it is \( 49 \times 49 \times 120 \).

## Other Mathematical Considerations

- **Parameter Count**: Calculating the number of parameters in each layer helps in understanding the model's complexity and memory requirements. For convolutional layers, the number of parameters is the product of the dimensions of the kernel, the number of input channels, and the number of output channels, plus the number of output channels if bias is used.
- **Feature Map Analysis**: Understanding how feature maps evolve through the network can provide insights into what the network is learning and how different layers contribute to the final decision-making process.
