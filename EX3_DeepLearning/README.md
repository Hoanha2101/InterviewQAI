# Detailed approach

### Data preparation:

+ Load and preprocess MNIST data.
+ Normalize the data and convert it to 2D.

### Model building:

+ Build a simple model with 2 layers: w1*X + b1 and w2*X + b2

### Triplet Loss:

+ Use Triplet Loss to train the model. Triplet Loss helps ensure that the distance between samples of the same class is smaller than the distance between samples of different classes.
+ Triplet Loss is calculated based on the Euclidean distance between the embeddings of the anchor, positive, and negative samples.

### Triplet Loss and Gradients

``The triplet loss function is given by:``

$$
\mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} \max\left(0, \|f(x_i^a) - f(x_i^p)\|^2 - \|f(x_i^a) - f(x_i^n)\|^2 + \alpha\right)
$$

where:
- $x_i^a$: is the output of the anchor. 
- $x_i^p$: is the output of the positive.
- $x_i^n$: is the output of the negative.
- $alpha$: is the margin.

The gradients of the triplet loss with respect to each output are computed as follows:

**Gradient with respect to anchor:**

$$
\frac{\partial \mathcal{L}}{\partial f(x_i^a)} = 2 \left( f(x_i^a) - f(x_i^p) \right) - 2 \left( f(x_i^a) - f(x_i^n) \right)
$$

**Gradient with respect to positive:**

$$
\frac{\partial \mathcal{L}}{\partial f(x_i^p)} = -2 \left( f(x_i^a) - f(x_i^p) \right)
$$

**Gradient with respect to negative:**

$$
\frac{\partial \mathcal{L}}{\partial f(x_i^n)} = 2 \left( f(x_i^a) - f(x_i^n) \right)
$$

### Model training:

+ Train the model with batches of data.
+ Use gradient descent to update the model weights.

### Update weight:

***Gradient and Weight Update Formulas with ``learning rate`` $\eta \cdot$***

``Gradient with respect to weights``

$$
\frac{\partial \mathcal{L}}{\partial W} = \text{input}^T \cdot \text{grad}_i
$$

``Gradient with respect to bias``

$$
\frac{\partial \mathcal{L}}{\partial b} = \text{grad}_i
$$

``Weight Update``

$$
W := W - \eta \cdot \frac{\partial \mathcal{L}}{\partial W}
$$

``Bias Update``

$$
b := b - \eta \cdot \frac{\partial \mathcal{L}}{\partial b}
$$

### Predict:
+ Use the output vector from the model to classify the image to be predicted to belong to which class.

### Save and load the model:

+ Save the model after training and load the model again for testing.

# Advantages and disadvantages of using Machine Learning

#### Advantages:
+ Simple and easy to implement: Traditional Machine Learning algorithms are often simpler and easier to implement.
+ Low resource requirements: No need for powerful computing resources such as GPUs.
+ Works well with small data: Can work well with smaller amounts of data, when the data is carefully designed and selected.

#### Disadvantages:
+ Need for hand-crafted features: Requires hand-crafted features, which can limit the ability to learn complex patterns.
+ Lower generalization ability: May not generalize as well as Deep Learning models when faced with complex data.