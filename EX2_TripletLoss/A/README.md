## A. Formula Triplet Loss with One Samples
The Triplet Loss can be expressed as:

$$\mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} \max\left(0, \|f(x_i^a) - f(x_i^p)\|^2 - \|f(x_i^a) - f(x_i^n)\|^2 + \alpha\right)$$

Where:
- $x_i^a$ is the anchor.
- $x_i^p$ is the positive example.
- $x_i^n$ is the negative example.
- $α$ is the margin.
- $N$ is the number of triplets used in the loss calculation.