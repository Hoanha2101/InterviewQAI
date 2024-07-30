## B. Formula Triplet Loss with Multiple Samples
$$\mathcal{L} = \frac{1}{A} \sum_{i=1}^{N} \max\left(0, \frac{1}{P} \sum_{p\in P}\|f(x_i^a) - f(x_i^p)\|^2 - \frac{1}{N}\sum_{n\in N} \|f(x_i^a) - f(x_i^n)\|^2 + \alpha\right)$$

Where:
- $x_i^a$ is the anchor.
- $x_i^p$ is the positive example.
- $x_i^n$ is the negative example.
- $α$ is the margin.
- $P$ is number of positive samples.
- $N$ is number of negative samples.
- $A$ is the number of triplets used in the loss calculation.