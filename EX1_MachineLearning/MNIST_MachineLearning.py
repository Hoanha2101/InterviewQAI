from lib import *
from utils import *

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize data
X_train = X_train / 255.0
X_test = X_test / 255.0

#Reshape the data to 2D
X_train = X_train.reshape(-1, 28*28)
X_test = X_test.reshape(-1, 28*28)

#Predict
# y_pred = predict(X_train[:100], y_train[:100], X_test, k=3)

y_pred = predict(X_train, y_train, X_test, k=3)

#Cal accuracy
accuracy = np.sum(y_pred == y_test) / len(y_test)

print(f'Accuracy: {accuracy}')
