import numpy as np
from lib import *
epoch = 10
batch_size = 64
learning_rate = 0.01

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

def train(model, X_train, y_train, epochs=10, batch_size=64, learning_rate=0.01):
    num_samples = X_train.shape[0]
    for epoch in range(epochs):
        indices = np.random.permutation(num_samples)
        for i in range(0, num_samples, batch_size):
            end = i + batch_size
            if end > num_samples:
                end = num_samples
            batch_indices = indices[i:end]
            anchors = X_train[batch_indices]
            anchor_labels = y_train[batch_indices]

            # Choose positive examples from the same class
            positives = np.empty_like(anchors)
            for j, anchor_label in enumerate(anchor_labels):
                positive_indices = np.where((y_train == anchor_label).all(axis=1))[0]
                positive_indices = positive_indices[positive_indices != batch_indices[j]]
                positive_index = np.random.choice(positive_indices)
                positives[j] = X_train[positive_index]

            # Ensure negatives are different from anchors
            negatives = np.empty_like(anchors)
            for j, anchor_label in enumerate(anchor_labels):
                negative_indices = np.where((y_train != anchor_label).any(axis=1))[0]
                negative_index = np.random.choice(negative_indices)
                negatives[j] = X_train[negative_index]

            # Forward pass
            anchor_out = model.forward(anchors)
            positive_out = model.forward(positives)
            negative_out = model.forward(negatives)

            # Compute triplet loss
            if anchor_out.shape[0] == positive_out.shape[0] == negative_out.shape[0]:
                loss = triplet_loss(anchor_out, positive_out, negative_out)
                print(f'Epoch {epoch+1}, Batch {i//batch_size+1}, Loss: {loss}')

                # Backward pass and update weights
                grad_output = 2 * (anchor_out - positive_out) - 2 * (anchor_out - negative_out)
                model.backward(anchors, grad_output, learning_rate)
            else:
                print("Output dimensions mismatch")

def classify_image(model, image, class_vectors):
    image_vector = model.forward(image.reshape(1, -1))
    distances = np.linalg.norm(class_vectors - image_vector, axis=1)
    predicted_class = np.argmin(distances)
    return predicted_class

def save_class_vectors(class_vectors, filename):
    with open(filename, 'wb') as f:
        pickle.dump(class_vectors, f)

def load_class_vectors(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
    
def preprocess(image):
    image = image / 255.0
    image = image.reshape(1, -1)
    return image

def predict(model, image, class_vectors):
    image_vector = model.forward(preprocess(image))
    sim = []
    for i in range(len(class_vectors)):
        vector_org = class_vectors[i]
        sim.append(cosine_similarity(image_vector, vector_org)[0][0])
    return np.argmax(sim)