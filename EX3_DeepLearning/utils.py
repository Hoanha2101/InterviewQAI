import numpy as np
from lib import *
epoch = 10
batch_size = 64

def triplet_loss(anchor, positive, negative, margin=1.0):
    pos_dist = np.sum((anchor - positive) ** 2, axis=1)
    neg_dist = np.sum((anchor - negative) ** 2, axis=1)
    loss = np.maximum(0, pos_dist - neg_dist + margin)
    return np.mean(loss)

def train(model, X_train, y_train, epochs=epoch, batch_size=batch_size):
    num_samples = X_train.shape[0]
    for epoch in range(epochs):
        indices = np.random.permutation(num_samples)
        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i:i+batch_size]
            anchors = X_train[batch_indices]
            
            # Choose positive examples from the same batch
            positives = anchors.copy()
            
            # Ensure negatives are different from anchors
            negative_indices = np.random.choice(num_samples, batch_size, replace=False)
            negatives = X_train[negative_indices]
            
            # Forward pass
            anchor_out = model.forward(anchors)
            positive_out = model.forward(positives)
            negative_out = model.forward(negatives)
            
            # Compute triplet loss
            # Ensure the output size is correct
            if anchor_out.shape[0] == positive_out.shape[0] == negative_out.shape[0]:
                loss = triplet_loss(anchor_out, positive_out, negative_out)
                print(f'Epoch {epoch+1}, Loss: {loss}')
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
    print(preprocess(image))
    image_vector = model.forward(preprocess(image))
    sim = []
    for i in range(len(class_vectors)):
        vector_org = class_vectors[i]
        sim.append(cosine_similarity(image_vector, vector_org)[0][0])
    return np.argmax(sim)