from lib import *
from utils import *
from model import SimpleNN
from sklearn.preprocessing import LabelBinarizer

# Load MNIST data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize data
X_train = X_train / 255.0
X_test = X_test / 255.0

# Reshape the data to 2D
X_train = X_train.reshape(-1, 28*28)
X_test = X_test.reshape(-1, 28*28)

# Reduce data size for quick testing
X_train = X_train[:1000]
X_test = X_test[:1000]
y_train = y_train[:1000]
y_test = y_test[:1000]

# Convert labels to one-hot encoding
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.transform(y_test)

print(f'y_train shape after binarization: {y_train.shape}')

# Initialize and train the model
input_dim = 28*28
output_dim = 100  # Number of classes in MNIST dataset
model = SimpleNN(input_dim, output_dim)
train(model, X_train, y_train)

# Save model weights
model.save('EX3_DeepLearning/weights/model.pkl')

# Choose 10 representative images for each class
class_indices = [np.where(y_train[:, i] == 1)[0][0] for i in range(10)]
class_images = X_train[class_indices]
# Generate class vectors
class_vectors = np.array([model.forward(image.reshape(1, -1)) for image in class_images])
# Save class vectors
save_class_vectors(class_vectors, 'EX3_DeepLearning/weights/class_vectors.pkl')

print("Saved")

# model.load("EX3_DeepLearning/weights/model.pkl")