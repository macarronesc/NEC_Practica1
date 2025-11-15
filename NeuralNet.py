import numpy as np

class NeuralNet:
  def __init__(self, layers, learning_rate=0.01, momentum=0.1, epochs=100, activation='sigmoid'):
    # Hiperparams
    self.learning_rate = learning_rate
    self.momentum = momentum
    self.epochs = epochs

    # Architecture of the network
    self.L = len(layers)      # Num of layers
    self.n = layers.copy()    # Array with the number of units in each layer

    # Activation function configuration
    self.fact_name = activation
    if self.fact_name == 'sigmoid':
      self.fact = lambda x: 1 / (1 + np.exp(-x))
      self.fact_derivative = lambda y: y * (1 - y)
    # TODO: Add other activation functions like relu, tanh, linear
    
    # Initialization of all network variables according to the specification
    self._initialize_network_variables()
    
    # Storage for error evolution (not used yet)
    self.train_loss_history = []
    self.val_loss_history = []

  def _initialize_network_variables(self):
    # Activations (xi) and Fields (h)
    self.xi = [np.zeros((n_units, 1)) for n_units in self.n]
    # h[0] is not used, a placeholder is added to maintain index consistency
    self.h = [np.zeros((1,1))] + [np.zeros((n_units, 1)) for n_units in self.n[1:]]

    # Weights (w) and Thresholds (theta) - Using Xavier/Glorot initialization
    # w[0] and theta[0] are not used, placeholders are added
    self.w = [np.zeros((1,1))]
    self.theta = [np.zeros((1,1))] 
    for l in range(1, self.L):
      # Weights are initialized with small random values to break symmetry
      limit = np.sqrt(6 / (self.n[l-1] + self.n[l]))
      self.w.append(np.random.uniform(-limit, limit, (self.n[l], self.n[l-1])))
      self.theta.append(np.random.uniform(-limit, limit, (self.n[l], 1)))

    # Error propagation (delta)
    self.delta = [np.zeros((1,1))] + [np.zeros((n_units, 1)) for n_units in self.n[1:]]

    # Changes for weights (d_w) and thresholds (d_theta) for the momentum term
    self.d_w = [np.zeros_like(w_matrix) for w_matrix in self.w]
    self.d_theta = [np.zeros_like(theta_vec) for theta_vec in self.theta]
    self.d_w_prev = [np.zeros_like(w_matrix) for w_matrix in self.w]
    self.d_theta_prev = [np.zeros_like(theta_vec) for theta_vec in self.theta]

  def _feed_forward(self, x):
    # The input x is set as the activation of the first layer (layer 0)
    self.xi[0] = x.reshape(-1, 1)
    
    # Propagation through hidden and output layers
    for l in range(1, self.L):
      # Field: dot product of weights with activation from previous layer, minus the threshold
      self.h[l] = self.w[l] @ self.xi[l-1] - self.theta[l]
      # Activation: apply the activation function to the field
      self.xi[l] = self.fact(self.h[l])
      
    # Returns the activation of the last layer (the prediction)
    return self.xi[self.L - 1]

  def _back_propagate(self, y):
    # TODO: Implement the calculation of deltas for each layer, starting from the last.
    pass
  
  def _update_weights(self):
    # TODO: Implement the update of w and theta using learning_rate and momentum.
    pass

  def fit(self, X, y):
    print("The training method 'fit' is not yet implemented.")
    # TODO: Implement the main training loop over epochs,
    # iterating over samples, calling feed_forward, back_propagate, and update_weights.
    pass

  def predict(self, X):
    if X.ndim == 1:
      # If it is a single vector, reshape it to have 2 dimensions
      X = X.reshape(1, -1)
    
    # Apply feed_forward to each input sample
    predictions = np.array([self._feed_forward(x) for x in X])
    
    # Flatten the result to return a simple prediction vector
    return predictions.flatten()


# --- Test code for the intermediate version ---
layers = [4, 9, 5, 1]
# Now the constructor requires hyperparameters, even if they are not yet used for training
nn = NeuralNet(layers, learning_rate=0.05, epochs=100)

print("Network architecture:")
print("L (number of layers) = ", nn.L)
print("n (units per layer) = ", nn.n)
print("-" * 20)

print("Dimensions of weights (w):")
for i in range(1, nn.L):
  print(f"  w[{i}]: {nn.w[i].shape}")

print("Dimensions of thresholds (theta):")
for i in range(1, nn.L):
  print(f"  theta[{i}]: {nn.theta[i].shape}")
print("-" * 20)

# Create a test input sample (4 features)
x_test = np.random.rand(4)
print("Test input (x_test):", x_test)

# The network can now predict, although the weights are random
prediction = nn.predict(x_test)
print("Prediction (with random weights):", prediction)

# Call fit to show that it is not yet implemented
nn.fit(None, None)