import numpy as np

def triplet_loss(anchor, positive, negative, alpha=0.2):
    pos_dist = np.sum((anchor - positive) ** 2, axis=1)
    neg_dist = np.sum((anchor - negative) ** 2, axis=1)
    loss = np.maximum(0, pos_dist - neg_dist + alpha)
    return np.mean(loss)

# Giả sử các embedding có dạng như sau
anchor = np.array([[0.5, 0.1], [0.3, 0.2]])
positive = np.array([[0.6, 0.1], [0.4, 0.2]])
negative = np.array([[1.0, 1.0], [0.9, 0.8]])

# Tính toán triplet loss
loss = triplet_loss(anchor, positive, negative, alpha=1)
print("Triplet Loss:", loss)
