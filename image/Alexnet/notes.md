
# Summary of "ImageNet Classification with Deep Convolutional Neural Networks"

## Authors
- Alex Krizhevsky
- Ilya Sutskever
- Geoffrey E. Hinton

## Abstract
- The paper discusses training a deep convolutional neural network for classifying high-resolution images in the ImageNet LSVRC-2010 contest into 1000 different classes.
- Achieved top-1 and top-5 error rates significantly better than the previous state-of-the-art.

## Key Concepts
1. **Deep Convolutional Neural Network (CNN):** Utilized a large CNN with 60 million parameters and 650,000 neurons, consisting of five convolutional layers, max-pooling layers, and three fully-connected layers.
2. **Efficient Training Techniques:**
   - Non-saturating neurons.
   - Efficient GPU implementation of the convolution operation.
3. **Dropout for Overfitting Reduction:** A regularization method that randomly omits a subset of features at each training stage.
4. **Data Augmentation:** Used for combating overfitting, involves generating image translations and alterations to intensities of RGB channels.
5. **ReLU Nonlinearity:** Demonstrated that deep CNNs with ReLUs (Rectified Linear Units) train several times faster than their equivalents with tanh units.
6. **Multiple GPUs Training:** The model was trained using two GPUs to manage the large computational demand.
7. **Local Response Normalization:** Implemented a form of lateral inhibition inspired by real neurons, which was found to aid in generalization.
8. **Overlapping Pooling:** Used pooling layers that overlap to reduce the top-1 and top-5 error rates.


## ReLU (Rectified Linear Unit)
- **Equation:** `f(x) = max(0, x)`
- **Description:** ReLU is a type of activation function used in neural networks. It replaces negative pixel values in the feature map with zero. This nonlinearity is simple and effective, helping with the vanishing gradient problem and speeding up training.
- **Notes**
  - It is faster than tanh

## Local Response Normalization (LRN)
- **Equation:** 
  \[ b^i_{x,y} = \frac{a^i_{x,y}}{\left(k + \alpha \sum_{j=\max(0, i - \frac{n}{2})}^{\min(N-1, i + \frac{n}{2})} (a^j_{x,y})^2\right)^\beta} \]
  - Where \( a_{x,y}^i \) is the activity of a neuron computed by applying kernel `i` at position `(x, y)` and then applying the ReLU nonlinearity.
  - The sum runs over `n` adjacent kernel maps at the same spatial position, and `N` is the total number of kernels in the layer.
- **Description:** LRN performs a kind of lateral inhibition inspired by the activity of neurons in the brain, where the excitatory responses of a neuron are normalized by the activity of its neighbors. This normalization is done across the channels, not the batch dimension.

- **Notes**
   - constants used in paper: k=2, n=5, a=10^-4, B=0.75
   - applied after the relu on certain layers


## Overlapping Pooling
- **Notes**
   - pooling layers are used to summarized the outputs of the neighboring groups if neurons 
   in the same kernel map.
   -  a pooling layer can be thought of as consisting of a grid of pooling units spaced s pixels apart, each summarizing a neighborhood of size z × z centered at the location of the pooling unit.
   -  if we set s = z then we do normal trad. pooling
   -  if we set s < z then we get overlapping pooling
   -  they use s=2 and z=3 reduces error rate by 0.4%
   -  z stands for kernel size and s stands for stride when thinking of pytorch layer
   -  this helps it to not overfit 

   - modifed way of calculating pooling output size when they overlap : \[ \text{Output Size}= \left( \frac{\text{Input Size} - \text{Pool Size}}{\text{Stride}} \right) + 1 \]
