import numpy as np

class LogisticRegressionScratch:
    def __init__(self, learning_rate=0.001, iterations=100000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None
    
    # Sigmoid function
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    # Fitter modellen med gradient descent
    def fit(self, X, y):
        # Antall samples (m) og antall features (n)
        m, n = X.shape
        
        # Initialiserer vekter og bias (0)
        self.weights = np.zeros(n)
        self.bias = 0
        
        # Gradient descent
        for _ in range(self.iterations):
            # Lineær kombinasjon av vekter og features (z = w * X + b)
            z = np.dot(X, self.weights) + self.bias
            
            # benytter  sigmoid funksjonen for predicted probabilities
            y_pred = self.sigmoid(z)
            
            # Compute the gradients (derivatives)
            dw = (1 / m) * np.dot(X.T, (y_pred - y))  # Gradient of weights
            db = (1 / m) * np.sum(y_pred - y)         # Gradient of bias
            
            # Oppdaterer vektene og bias ved bruk av gradients
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict_proba(self, X):
        z = np.dot(X, self.weights) + self.bias
        y_prob = self.sigmoid(z)
        # returnerer sannsynligheter for begge klassene [P(class=0), P(class=1)]
        return np.vstack([1 - y_prob, y_prob]).T

    # Predikerer  labels basert på vektene den lærte
    def predict(self, X):
        # Lineær kombinasjon og bruk av sigmoid for å få sannsynligheter
        z = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(z)
        
        # konverterer sannsynliigheter til binært (0 eller 1) basert på threshold av 0.5
        return np.where(y_pred > 0.5, 1, 0)