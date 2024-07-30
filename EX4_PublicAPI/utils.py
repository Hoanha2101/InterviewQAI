import numpy as np
import pickle
from model import SimpleNN

def triplet_loss(anchor, positive, negative, alpha=0.2):
    pos_dist = np.sum((anchor - positive) ** 2, axis=1)
    neg_dist = np.sum((anchor - negative) ** 2, axis=1)
    loss = np.maximum(0, pos_dist - neg_dist + alpha)
    return np.mean(loss)

def load_class_vectors(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
    
def preprocess(image):
    image = image / 255.0
    image = image.reshape(1, -1)
    return image

input_dim = 28*28
output_dim = 100  # Number of classes in MNIST dataset
model = SimpleNN(input_dim, output_dim)
model.load("EX4_PublicAPI/weights/model.pkl")

loaded_class_vectors = load_class_vectors('EX4_PublicAPI/weights/class_vectors.pkl')

def predict(model, image, class_vectors):
    print(preprocess(image))
    image_vector = model.forward(preprocess(image))
    sim = []
    for i in range(len(class_vectors)):
        vector_org = class_vectors[i]
        sim.append(cosine_similarity(image_vector, vector_org)[0][0])
    return np.argmax(sim)
