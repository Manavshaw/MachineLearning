import numpy as np

class LinearRegression: 
    def __init__(self, learning_rate = 0.001, iterations = 100000): 
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.theta_0 = 0
        self.theta_1 = 0
    
    def fit (self, x, y): 
        X = np.array(x)
        Y = np.array(y)

        if len(X) != len(Y):
            raise ValueError("Input arrays X and Y must be of the same length.")
        m = len(y)
        
        self.theta_0 = 0
        self.theta_1 = 0

        for _ in range(self.iterations):
            y_pred = self.theta_0 + self.theta_1 * X
            
            d_theta_0 = (1/m) * np.sum(y_pred - Y)
            d_theta_1 = (1/m) * np.sum((y_pred - Y) * X)
            
            self.theta_0 -= self.learning_rate * d_theta_0
            self.theta_1 -= self.learning_rate * d_theta_1
        
    def predict(self, x):
        X = np.array(x)
        return self.theta_0 + self.theta_1 * X