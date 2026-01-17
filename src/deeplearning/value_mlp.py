"""
Scalar-based MLP using Value Class (Autograd)
==============================================

This module implements a Multi-Layer Perceptron using the Value class
for automatic differentiation. Unlike the vectorized numpy implementation,
this version operates on individual scalar values, demonstrating the power
of our autograd system.

Architecture
------------
The network is built from individual neurons that use Value objects:
    
    Input (784) → Dense(784, 128) → ReLU → Dense(128, 10) → Softmax

Each neuron computes: activation(Σ(w_i * x_i) + b)

This implementation is educational and demonstrates:
- How autograd enables building neural networks from scratch
- The connection between scalar operations and neural networks
- How backpropagation works at the individual neuron level

Note: This is slower than vectorized implementations but more transparent.

Author: Deep Learning from Scratch
"""

import numpy as np
from typing import List
from deeplearning.autograd import Value
import random


class Neuron:
    """
    A single neuron with scalar-based autograd.
    
    A neuron computes: activation(Σ(w_i * x_i) + b)
    
    Parameters
    ----------
    n_inputs : int
        Number of input connections
    activation : str
        Activation function: 'relu', 'tanh', 'sigmoid', or 'linear'
        
    Attributes
    ----------
    weights : List[Value]
        Weight values for each input
    bias : Value
        Bias term
    activation : str
        Activation function name
    """
    
    def __init__(self, n_inputs: int, activation: str = 'relu'):
        # Initialize weights with small random values
        self.weights = [Value(random.uniform(-1, 1) * np.sqrt(2.0 / n_inputs)) 
                        for _ in range(n_inputs)]
        self.bias = Value(0.0)
        self.activation = activation
    
    def __call__(self, x: List[Value]) -> Value:
        """
        Forward pass through the neuron.
        
        Parameters
        ----------
        x : List[Value]
            Input values
            
        Returns
        -------
        Value
            Output after activation
        """
        # Compute weighted sum: Σ(w_i * x_i) + b
        out = sum((wi * xi for wi, xi in zip(self.weights, x)), self.bias)
        
        # Apply activation
        if self.activation == 'relu':
            return out.relu()
        elif self.activation == 'tanh':
            return out.tanh()
        elif self.activation == 'sigmoid':
            return out.sigmoid()
        else:  # linear
            return out
    
    def parameters(self) -> List[Value]:
        """Return all parameters (weights and bias)."""
        return self.weights + [self.bias]
    
    def zero_grad(self):
        """Reset gradients for all parameters."""
        for p in self.parameters():
            p.zero_grad()


class Layer:
    """
    A layer of neurons.
    
    Parameters
    ----------
    n_inputs : int
        Number of inputs to each neuron
    n_outputs : int
        Number of neurons in the layer
    activation : str
        Activation function for all neurons
        
    Attributes
    ----------
    neurons : List[Neuron]
        List of neurons in this layer
    """
    
    def __init__(self, n_inputs: int, n_outputs: int, activation: str = 'relu'):
        self.neurons = [Neuron(n_inputs, activation) for _ in range(n_outputs)]
    
    def __call__(self, x: List[Value]) -> List[Value]:
        """
        Forward pass through the layer.
        
        Parameters
        ----------
        x : List[Value]
            Input values
            
        Returns
        -------
        List[Value]
            Output from each neuron
        """
        return [neuron(x) for neuron in self.neurons]
    
    def parameters(self) -> List[Value]:
        """Return all parameters from all neurons."""
        return [p for neuron in self.neurons for p in neuron.parameters()]
    
    def zero_grad(self):
        """Reset gradients for all neurons."""
        for neuron in self.neurons:
            neuron.zero_grad()


class ValueMLP:
    """
    Multi-Layer Perceptron using Value-based autograd.
    
    This is a scalar-based implementation that demonstrates automatic
    differentiation at the individual neuron level.
    
    Parameters
    ----------
    layer_sizes : List[int]
        List of layer sizes including input size
        Example: [784, 128, 10] creates:
            - Input: 784 features
            - Hidden: 128 neurons
            - Output: 10 neurons
    
    Attributes
    ----------
    layers : List[Layer]
        List of layers in the network
        
    Examples
    --------
    >>> mlp = ValueMLP([784, 128, 10])
    >>> # Create input (as list of Values)
    >>> x = [Value(v) for v in np.random.randn(784)]
    >>> # Forward pass
    >>> output = mlp(x)
    >>> # Backward pass
    >>> loss = output[0]  # Simplified
    >>> loss.backward()
    """
    
    def __init__(self, layer_sizes: List[int]):
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            # Last layer uses linear activation (logits)
            activation = 'relu' if i < len(layer_sizes) - 2 else 'linear'
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i + 1], activation))
    
    def __call__(self, x: List[Value]) -> List[Value]:
        """
        Forward pass through all layers.
        
        Parameters
        ----------
        x : List[Value]
            Input values
            
        Returns
        -------
        List[Value]
            Output logits
        """
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self) -> List[Value]:
        """Return all parameters from all layers."""
        return [p for layer in self.layers for p in layer.parameters()]
    
    def zero_grad(self):
        """Reset gradients for all parameters."""
        for layer in self.layers:
            layer.zero_grad()
    
    def num_parameters(self) -> int:
        """Return total number of parameters."""
        return len(self.parameters())


def softmax_cross_entropy_loss(logits: List[Value], target: int) -> Value:
    """
    Compute softmax cross-entropy loss for a single sample.
    
    Loss = -log(exp(logit_target) / Σ(exp(logit_i)))
         = -logit_target + log(Σ(exp(logit_i)))
    
    Parameters
    ----------
    logits : List[Value]
        Raw output scores from network
    target : int
        Target class index
        
    Returns
    -------
    Value
        Scalar loss value
    """
    # Compute log-sum-exp for numerical stability
    # max_logit = max(l.data for l in logits)
    
    # Compute sum of exponentials
    exp_sum = sum((logit.exp() for logit in logits), Value(0.0))
    
    # Cross-entropy loss
    loss = exp_sum.log() - logits[target]
    
    return loss


def train_step_value_mlp(
    model: ValueMLP,
    x_batch: np.ndarray,
    y_batch: np.ndarray,
    lr: float
) -> float:
    """
    Perform one training step on a batch.
    
    Note: This processes samples one at a time (no batching at Value level).
    
    Parameters
    ----------
    model : ValueMLP
        The neural network
    x_batch : np.ndarray
        Input batch of shape (N, input_size)
    y_batch : np.ndarray
        Target labels of shape (N,)
    lr : float
        Learning rate
        
    Returns
    -------
    float
        Average loss over the batch
    """
    total_loss = 0.0
    batch_size = len(x_batch)
    
    for x_sample, y_sample in zip(x_batch, y_batch):
        # Zero gradients
        model.zero_grad()
        
        # Convert input to Values
        x_values = [Value(float(v)) for v in x_sample]
        
        # Forward pass
        logits = model(x_values)
        
        # Compute loss
        loss = softmax_cross_entropy_loss(logits, int(y_sample))
        
        # Backward pass
        loss.backward()
        
        # Update parameters (SGD)
        for p in model.parameters():
            p.data -= lr * p.grad
        
        total_loss += loss.data
    
    return total_loss / batch_size


def predict_value_mlp(model: ValueMLP, x: np.ndarray) -> np.ndarray:
    """
    Make predictions for a batch of inputs.
    
    Parameters
    ----------
    model : ValueMLP
        The neural network
    x : np.ndarray
        Input batch of shape (N, input_size)
        
    Returns
    -------
    np.ndarray
        Predicted class labels of shape (N,)
    """
    predictions = []
    
    for x_sample in x:
        # Convert to Values
        x_values = [Value(float(v)) for v in x_sample]
        
        # Forward pass
        logits = model(x_values)
        
        # Get prediction (argmax)
        logit_values = [l.data for l in logits]
        pred = np.argmax(logit_values)
        predictions.append(pred)
    
    return np.array(predictions)


def evaluate_value_mlp(model: ValueMLP, x: np.ndarray, y: np.ndarray) -> float:
    """
    Evaluate accuracy on a dataset.
    
    Parameters
    ----------
    model : ValueMLP
        The neural network
    x : np.ndarray
        Input data of shape (N, input_size)
    y : np.ndarray
        True labels of shape (N,)
        
    Returns
    -------
    float
        Accuracy as a fraction in [0, 1]
    """
    predictions = predict_value_mlp(model, x)
    return np.mean(predictions == y)


if __name__ == "__main__":
    """
    Demonstrate the Value-based MLP with a tiny example.
    """
    print("=" * 60)
    print("VALUE-BASED MLP DEMONSTRATION")
    print("=" * 60)
    
    # Set random seed
    random.seed(42)
    np.random.seed(42)
    
    # Create a tiny MLP
    print("\nCreating MLP with architecture: [4, 8, 3]")
    mlp = ValueMLP([4, 8, 3])
    print(f"Total parameters: {mlp.num_parameters()}")
    
    # Create tiny dataset
    print("\nGenerating tiny dataset...")
    x_train = np.random.randn(20, 4)
    y_train = np.random.randint(0, 3, 20)
    
    # Training
    print("\nTraining for 10 iterations...")
    for i in range(10):
        loss = train_step_value_mlp(mlp, x_train, y_train, lr=0.01)
        if i % 2 == 0:
            acc = evaluate_value_mlp(mlp, x_train, y_train)
            print(f"Iteration {i}: Loss = {loss:.4f}, Accuracy = {acc * 100:.1f}%")
    
    # Final evaluation
    print("\nFinal evaluation:")
    final_acc = evaluate_value_mlp(mlp, x_train, y_train)
    print(f"Training accuracy: {final_acc * 100:.1f}%")
    
    print("\n" + "=" * 60)
    print("Note: This is a scalar-based implementation for educational")
    print("purposes. For MNIST, use the vectorized implementation or")
    print("train on a small subset due to computational cost.")
    print("=" * 60)
