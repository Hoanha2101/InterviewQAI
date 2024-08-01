import numpy as np
import pickle
from model import SimpleNN
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

def triplet_loss(anchor, positive, negative, alpha=0.2):
    """
    Compute the triplet loss.

    Parameters:
    - anchor: Embeddings for the anchor samples.
    - positive: Embeddings for the positive samples.
    - negative: Embeddings for the negative samples.
    - alpha: Margin for the triplet loss.

    Returns:
    - Average triplet loss across all samples.
    """
    # Compute squared Euclidean distances
    pos_dist = np.sum((anchor - positive) ** 2, axis=1)
    neg_dist = np.sum((anchor - negative) ** 2, axis=1)
    
    # Compute the triplet loss
    loss = np.maximum(0, pos_dist - neg_dist + alpha)
    return np.mean(loss)

def load_class_vectors(filename):
    """
    Load class vectors from a file.

    Parameters:
    - filename: Path to the file containing the class vectors.

    Returns:
    - Class vectors as a numpy array.
    """
    with open(filename, 'rb') as f:
        return pickle.load(f)
    
def preprocess(image):
    """
    Preprocess the input image.

    Parameters:
    - image: Input image as a numpy array or PIL image.

    Returns:
    - Preprocessed image as a numpy array with shape (1, -1).
    """
    # Convert numpy array to PIL Image if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray((image * 255).astype(np.uint8))
        
    # Resize to 28x28 pixels and convert to grayscale
    image = image.resize((28, 28)).convert('L')
    
    # Normalize and reshape
    image = np.array(image) / 255.0
    return image.reshape(1, -1)

# Define dimensions for the neural network
input_dim = 28 * 28
output_dim = 100

# Initialize and load the model
model = SimpleNN(input_dim, output_dim)
model.load("EX4_PublicAPI/weights/model.pkl")

# Load precomputed class vectors
loaded_class_vectors = load_class_vectors('EX4_PublicAPI/weights/class_vectors.pkl')

def predict(model, image, class_vectors):
    """
    Predict the class of the input image using the k-NN approach with precomputed class vectors.

    Parameters:
    - model: Instance of SimpleNN model.
    - image: Input image for which to make the prediction.
    - class_vectors: Precomputed class vectors for each class.

    Returns:
    - Predicted class index.
    """
    # Preprocess the input image
    preprocessed_image = preprocess(image)
    
    # Get the image vector from the model
    image_vector = model.forward(preprocessed_image)
    
    # Compute similarity between the image vector and class vectors
    sim = []
    for i in range(len(class_vectors)):
        vector_org = class_vectors[i]
        sim.append(cosine_similarity(image_vector, vector_org)[0][0])
    
    # Return the index of the class with the highest similarity
    return np.argmax(sim)

