from model import SimpleNN
from lib import *
from utils import *

# Load MNIST data
(_,__), (X_test, y_test) = mnist.load_data()

# Normalize data
X_test = X_test / 255.0

# Reshape the data to 2D
X_test = X_test.reshape(-1, 28*28)

input_dim = 28*28
output_dim = 100  # Number of classes in MNIST dataset
model = SimpleNN(input_dim, output_dim)
model.load("EX3_DeepLearning/weights/model.pkl")

loaded_class_vectors = load_class_vectors('EX3_DeepLearning/weights/class_vectors.pkl')
sample_image = X_test[0]

predicted_class = predict(model, sample_image, loaded_class_vectors)
print(f'Predicted Class: {predicted_class}')