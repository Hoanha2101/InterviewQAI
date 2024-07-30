import numpy as np

def extended_triplet_loss(anchor, positives, negatives, alpha=0.2):
    
    pos_dist = np.mean([np.sum((anchor - positive) ** 2, axis=1) for positive in positives], axis=0)
    neg_dist = np.mean([np.sum((anchor - negative) ** 2, axis=1) for negative in negatives], axis=0)
    loss = np.maximum(0, pos_dist - neg_dist + alpha)
    return np.mean(loss)

# example
anchor = np.array([[0.5, 0.1], [0.3, 0.2]])
positives = [np.array([[0.6, 0.1], [0.4, 0.2]]), np.array([[0.5, 0.2], [0.3, 0.3]])]
negatives = [
    np.array([[1.0, 1.0], [0.9, 0.8]]),
    np.array([[1.1, 1.1], [0.8, 0.7]]),
    np.array([[1.2, 1.2], [0.7, 0.6]]),
    np.array([[1.3, 1.3], [0.6, 0.5]]),
    np.array([[1.4, 1.4], [0.5, 0.4]])
]

loss = extended_triplet_loss(anchor, positives, negatives,alpha=0.5)
print("Extended Triplet Loss:", loss)
