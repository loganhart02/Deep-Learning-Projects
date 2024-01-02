# Paper notes
the notes file that will contain all my notes from new things outlined in each paper

## Authors
- Karen Simonyan
- Andrew Zisserman

## Abstract
- Investigates convolutional network depth's effect on accuracy in large-scale image recognition.
- Significant improvement achieved by increasing depth to 16-19 weight layers.
- Team secured first and second places in the ImageNet Challenge 2014 in localization and classification tracks respectively.
- Demonstrates that deep visual representations generalize well to other datasets.
- Two best-performing ConvNet models made publicly available.

## Key Concepts
1. **Architecture:** 
   - Input: Fixed-size 224x224 RGB images.
   - Preprocessing: Subtracting mean RGB value computed on the training set.
   - Convolutional Layers: Uses 3x3 filters (smallest size to capture spatial orientation).
   - Configuration Variants: Configurations A-E with depth varying from 11 to 19 weight layers.
   - Fully-Connected Layers: Three layers, the first two with 4096 channels each, the third for 1000-way ILSVRC classification.
   - Rectification: All hidden layers use ReLU non-linearity.
   - No Local Response Normalisation (LRN) except in one network due to increased memory and computation time.

2. **Training:**
   - Follows Krizhevsky et al. (2012) but with modifications.
   - Mini-batch gradient descent with momentum.
   - Regularisation: Weight decay and dropout for the first two fully-connected layers.
   - Learning rate: Initially set to 10^-2, decreased by a factor of 10 upon validation set performance plateau.
   - Epochs: Training concluded after 370K iterations (74 epochs).

3. **Initialisation:**
   - Initialised first four convolutional layers and last three fully-connected layers with those of net A.
   - Weights for new layers randomly initialised.
   - Found that random initialisation without pre-training is feasible.

4. **Data Augmentation:**
   - Random cropping from rescaled training images.
   - Random horizontal flipping and RGB colour shift.

5. **Training Image Size (S):**
   - Two approaches: Fixed S (single-scale) and multi-scale training.
   - Evaluated models trained at S = 256 and S = 384.
   - Multi-scale training by randomly sampling S from a range [Smin, Smax].

6. **Testing Methodology:**
   - Isotropic rescaling of input images to a predefined size (Q).
   - Conversion of fully-connected layers to convolutional layers for dense application over the image.
   - Class score map averaged for a fixed-size vector of class scores.
   - Test set augmented by horizontal flipping of images.

7. **Implementation:**
   - Based on Caffe toolbox.
   - Training and evaluation on multiple GPUs.
   - Multi-GPU training exploits data parallelism.

8. **Performance Evaluation:**
   - Evaluated on ILSVRC-2012 dataset.
   - Metrics: Top-1 and top-5 error rates.

9. **Single and Multi-scale Evaluation:**
   - Classification error decreases with increased ConvNet depth.
   - Scale jittering at training time leads to better results than fixed smallest side training.

10. **Multi-crop Evaluation:**
    - Multiple crops and dense evaluation techniques are complementary.
    - Combination of both techniques outperforms each individual method.

11. **ConvNet Fusion:**
    - Averaging soft-max class posteriors of several models.
    - Ensemble approach improves performance.

12. **Comparison with State-of-the-Art:**
    - Achieved competitive results in ILSVRC-2014.
    - Deep ConvNets significantly outperformed previous models.
    - Best performance achieved with two-model combination.

13. **Conclusion:**
    - Depth in visual representations is crucial for classification accuracy.
    - Conventional ConvNet architecture with increased depth achieves state-of-the-art performance.
    - Models generalize well to a wide range of tasks and datasets.


## training details
- batch size: `256`
- use SGD with momentum
- momentum: `0.9`
- weght decay: `5 * 1e-4`
- dropout regularization for first two fully connected layers
- dropout ratio: `0.5`
- image size: `224 x 224`
- starting learning rate: `1e-2`
- decrease the learning rate by a factor of 10 when validation accuracy stops improving
- in total the learning rate was decreased 3 times
  
# architecture
- image size: `224x224`
- filter receptive field: `3x3`
- also use `1x1` config for some layers: this makes them linear transformations
- conv stride: `1`
- padding: `1` for `3x3` conv layers
- ReLU activation is used
- 5 max pooling layers with `2x2` size and stride of `2`
- 3 total fully connected layers. first 2 have `4096` channels each. the last one is the final layer with 1000 channels
- they use LRN for one layer( side note this doesn't improve performance I'll add it to the implemntation but comment it out and not train with it)
- the LRN is the same as alexnet
- for my model I am only doing the 19 layer network
    - `16` conv layers
    - `3` FC layers
    - max pool is after first `2`, then after the 3rd and 4th, then after the 8th, then after the 12th, then after the 16th
    - softmax after last fc layer