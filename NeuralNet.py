import numpy as np

class NeuralNet:
  """
  A from-scratch implementation of a multilayer perceptron with backpropagation.

  This class adheres to the specifications provided in the NEC course assignment,
  including variable naming conventions (L, n, w, xi, etc.) and required methods.
  """
  
  def __init__(self, layers, learning_rate=0.01, momentum=0.1, epochs=100, activation='sigmoid', validation_split=0.2):
    """
    Initializes the Neural Network.

    Args:
      layers (list): A list of integers representing the number of units in each layer.
                     Example: [n_features, 10, 5, 1] for a network with 2 hidden layers.
      learning_rate (float): The learning rate for weight updates.
      momentum (float): The momentum term for weight updates.
      epochs (int): The number of training epochs.
      activation (str): The name of the activation function to use ('sigmoid', 'relu', 'tanh', 'linear').
      validation_split (float): The fraction of training data to be used as a validation set.
                                Set to 0 to disable validation.
    """
    # Hyperparameters
    self.learning_rate = learning_rate
    self.momentum = momentum
    self.epochs = epochs
    self.validation_split = validation_split

    # Network Structure
    self.L = len(layers)  # Number of layers
    self.n = layers.copy() # Array with number of units in each layer

    # Activation Function Setup
    self.fact_name = activation
    self.fact, self.fact_derivative = self._get_activation_functions(activation)

    # Initialize network variables as per assignment specification
    self._initialize_network_variables()
    
    # Storage for loss evolution
    self.train_loss_history = []
    self.val_loss_history = []


  def _get_activation_functions(self, name):
    """Returns the activation function and its derivative."""
    if name == 'sigmoid':
      return (lambda x: 1 / (1 + np.exp(-x))), \
             (lambda y: y * (1 - y))
    elif name == 'relu':
      # A small epsilon to prevent issues with division by zero in rare cases
      return (lambda x: np.maximum(0, x)), \
             (lambda y: np.where(y > 0, 1, 0))
    elif name == 'tanh':
      return (lambda x: np.tanh(x)), \
             (lambda y: 1 - y**2)
    elif name == 'linear':
      return (lambda x: x), \
             (lambda y: np.ones_like(y))
    else:
      raise ValueError(f"Activation function '{name}' is not supported.")

  def _initialize_network_variables(self):
    """Initializes all required variables for the network structure and backpropagation."""
    # Activations (xi) and Fields (h)
    self.xi = [np.zeros((n_units, 1)) for n_units in self.n]
    self.h = [np.zeros((1,1))] + [np.zeros((n_units, 1)) for n_units in self.n[1:]]

    # Weights (w) and Thresholds (theta) - Using Xavier/Glorot initialization
    self.w = [np.zeros((1,1))] # Placeholder for w[0]
    self.theta = [np.zeros((1,1))] # Placeholder for theta[0]
    for l in range(1, self.L):
      limit = np.sqrt(6 / (self.n[l-1] + self.n[l]))
      self.w.append(np.random.uniform(-limit, limit, (self.n[l], self.n[l-1])))
      self.theta.append(np.random.uniform(-limit, limit, (self.n[l], 1)))

    # Propagation of errors (delta)
    self.delta = [np.zeros((1,1))] + [np.zeros((n_units, 1)) for n_units in self.n[1:]]

    # Changes for weights (d_w) and thresholds (d_theta) for momentum
    self.d_w = [np.zeros_like(w_matrix) for w_matrix in self.w]
    self.d_theta = [np.zeros_like(theta_vec) for theta_vec in self.theta]
    self.d_w_prev = [np.zeros_like(w_matrix) for w_matrix in self.w]
    self.d_theta_prev = [np.zeros_like(theta_vec) for theta_vec in self.theta]

  def _feed_forward(self, x):
    """
    Performs a feed-forward pass for a single input sample.
    """
    self.xi[0] = x.reshape(-1, 1)
    for l in range(1, self.L):
      self.h[l] = self.w[l] @ self.xi[l-1] - self.theta[l]
      self.xi[l] = self.fact(self.h[l])
    return self.xi[self.L - 1]

  def _back_propagate(self, y):
    """
    Performs the backpropagation step for a single sample.
    """
    error = y - self.xi[self.L - 1]
    self.delta[self.L - 1] = error * self.fact_derivative(self.xi[self.L - 1])

    for l in range(self.L - 2, 0, -1):
      self.delta[l] = (self.w[l+1].T @ self.delta[l+1]) * self.fact_derivative(self.xi[l])
  
  def _update_weights(self):
    """Updates the weights and thresholds after a backpropagation step."""
    for l in range(1, self.L):
      self.d_w[l] = self.learning_rate * self.delta[l] @ self.xi[l-1].T
      self.d_theta[l] = -self.learning_rate * self.delta[l]

      self.w[l] += self.d_w[l] + self.momentum * self.d_w_prev[l]
      self.theta[l] += self.d_theta[l] + self.momentum * self.d_theta_prev[l]
      
      self.d_w_prev[l] = self.d_w[l]
      self.d_theta_prev[l] = self.d_theta[l]

  def fit(self, X, y):
    """
    Trains the network using the provided training data.
    """
    if self.validation_split > 0 and self.validation_split < 1:
      split_idx = int(len(X) * (1 - self.validation_split))
      X_train, X_val = X[:split_idx], X[split_idx:]
      y_train, y_val = y[:split_idx], y[split_idx:]
    else:
      X_train, y_train = X, y
      X_val, y_val = None, None

    for epoch in range(self.epochs):
      indices = np.arange(X_train.shape[0])
      np.random.shuffle(indices)
      X_train_shuffled, y_train_shuffled = X_train[indices], y_train[indices]

      epoch_train_error = 0
      for x_sample, y_sample in zip(X_train_shuffled, y_train_shuffled):
        prediction = self._feed_forward(x_sample)
        self._back_propagate(y_sample)
        self._update_weights()
        epoch_train_error += (y_sample - prediction)**2
      
      self.train_loss_history.append(epoch_train_error.item() / len(X_train))

      if X_val is not None:
        y_val_pred = self.predict(X_val)
        val_error = np.mean((y_val - y_val_pred)**2)
        self.val_loss_history.append(val_error)
        
    return self

  def predict(self, X):
    """
    Predicts the output for a given set of input samples.
    """
    if X.ndim == 1:
      X = X.reshape(1, -1)
    
    predictions = np.array([self._feed_forward(x) for x in X])
    return predictions.flatten()

  def loss_epochs(self):
    """
    Returns the training and validation loss history.
    """
    return np.array(self.train_loss_history), np.array(self.val_loss_history)

  def cross_validate(self, X, y, k=5):
    """
    Performs k-fold cross-validation. (Optional Part 2)
    """
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    fold_size = len(X) // k
    scores = []

    original_val_split = self.validation_split
    self.validation_split = 0 # Disable internal validation split during CV

    print(f"Starting {k}-fold Cross-Validation...")
    for i in range(k):
      start, end = i * fold_size, (i + 1) * fold_size
      val_idx = indices[start:end]
      train_idx = np.concatenate([indices[:start], indices[end:]])

      X_train, y_train = X[train_idx], y[train_idx]
      X_val, y_val = X[val_idx], y[val_idx]

      self._initialize_network_variables()
      self.fit(X_train, y_train)
      
      y_pred = self.predict(X_val)
      fold_mse = np.mean((y_val - y_pred)**2)
      scores.append(fold_mse)
      print(f"  Fold {i+1}/{k} - MSE: {fold_mse:.6f}")
    
    self.validation_split = original_val_split
    
    results = {'mean_mse': np.mean(scores), 'std_mse': np.std(scores)}
    print(f"CV finished. Mean MSE: {results['mean_mse']:.4f} (+/- {results['std_mse']:.4f})")
    return results