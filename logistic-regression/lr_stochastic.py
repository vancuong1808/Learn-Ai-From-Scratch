from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

class LogisticRegression:
    def __init__(self, input_size, num_classes, lr=0.01):
        self.W = np.zeros((input_size, num_classes))
        self.b = np.zeros((num_classes))
        self.lr = lr
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
        
    def predict(self, X):
        logits = np.dot(X, self.W) + self.b
        probs = self.sigmoid(logits)
        return probs.argmax(axis=1)

    def fit(self, X, y, epochs=20):
        for epoch in range(epochs):
            indices = np.random.permutation(X.shape[0])
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            for i in range(X_shuffled.shape[0]):
                xi = X_shuffled[i:i+1]
                yi = y_shuffled[i:i+1]
                yi_onehot = np.zeros((1, self.W.shape[1]))
                yi_onehot[0, yi] = 1
                
                logits = np.dot(xi, self.W) + self.b
                prediction = self.sigmoid(logits)
                
                error = prediction - yi_onehot
                
                # Gradient descent update
                self.W -= self.lr * np.dot(xi.T, error)
                self.b -= self.lr * error.sum(axis=0)

if __name__ == "__main__":
    data = load_iris()
    X, y = data.data, data.target
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=36)
    
    model = LogisticRegression(input_size=X.shape[1], num_classes=len(np.unique(y)), lr=0.1)
    model.fit(X_train, y_train, epochs=200)
    
    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions, target_names=data.target_names))
    
