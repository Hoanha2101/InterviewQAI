import numpy as np

def triplet_loss(anchor, positive, negative, alpha=0.3):
    '''
    Triplet loss using numpy
    Inputs:
        anchor: Numpy array of anchor embeddings
        positive: Numpy array of positive embeddings
        negative: Numpy array of negative embeddings
        alpha: Distance margin between positive and negative samples

    Returns:
        Computed loss
    '''

    positive_dist = np.sum(np.square(anchor - positive), axis=1)
    negative_dist = np.sum(np.square(anchor - negative), axis=1)

    loss_1 = positive_dist - negative_dist + alpha
    
    loss = np.sum(np.maximum(loss_1, 0.0))
    
    return loss

