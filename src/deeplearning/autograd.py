"""
Autograd Module
===============

This module implements automatic differentiation (autograd) for scalar values.
It provides a computational graph that tracks operations and computes gradients
via reverse-mode automatic differentiation (backpropagation).

The core class is Value, which wraps a scalar and builds a dynamic computational
graph as operations are performed. This enables automatic gradient computation
for arbitrary compositions of supported operations.

Supported Operations:
---------------------
- Arithmetic: +, -, *, /, **
- Activations: ReLU, tanh, sigmoid
- Mathematical: exp, log
- Unary: negation

Example:
--------
>>> from deeplearning.autograd import Value
>>> x = Value(2.0)
>>> y = Value(3.0)
>>> z = x * y + x ** 2
>>> z.backward()
>>> print(f"dz/dx = {x.grad}")  # Should be y + 2*x = 3 + 4 = 7
>>> print(f"dz/dy = {y.grad}")  # Should be x = 2
"""

from typing import Tuple, Set, List, Union
import numpy as np


class Value:
    """
    Scalar value in a computational graph with automatic differentiation.
    
    This class represents a node in a dynamic computational graph. Each Value
    stores:
    - data: the actual scalar value
    - grad: the gradient of some loss with respect to this value
    - _prev: parent nodes in the computation graph
    - _op: the operation that created this node
    - _backward: function to propagate gradients to parents
    
    The computational graph is built implicitly as operations are performed.
    Calling backward() on a Value computes gradients for all values in the
    graph that contributed to it.
    
    Parameters
    ----------
    data : float
        The scalar value to wrap
    _children : Tuple[Value, ...], optional
        Parent nodes that were used to compute this value
    _op : str, optional
        String describing the operation that created this value
        
    Attributes
    ----------
    data : float
        The scalar value
    grad : float
        Gradient of the loss with respect to this value (initialized to 0)
    _prev : Set[Value]
        Set of parent nodes in the computation graph
    _op : str
        Operation that created this node (for debugging/visualization)
    _backward : callable
        Function that propagates gradients to parent nodes
        
    Examples
    --------
    Basic arithmetic:
    >>> a = Value(2.0)
    >>> b = Value(3.0)
    >>> c = a * b + a ** 2
    >>> c.backward()
    >>> print(a.grad)  # dc/da = b + 2*a = 3 + 4 = 7
    7.0
    
    Neural network example:
    >>> x = Value(0.5)
    >>> w = Value(2.0)
    >>> b = Value(1.0)
    >>> y = (x * w + b).tanh()  # Simple neuron
    >>> y.backward()
    >>> print(w.grad)  # Gradient for weight update
    """

    def __init__(
        self, 
        data: float, 
        _children: Tuple['Value', ...] = (), 
        _op: str = ''
    ):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other: Union['Value', float]) -> 'Value':
        """
        Addition operation: self + other
        
        Gradient: d(a + b)/da = 1, d(a + b)/db = 1
        
        Parameters
        ----------
        other : Value or float
            Value to add (automatically wrapped if float)
            
        Returns
        -------
        Value
            Result of addition
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __mul__(self, other: Union['Value', float]) -> 'Value':
        """
        Multiplication operation: self * other
        
        Gradient: d(a * b)/da = b, d(a * b)/db = a
        
        Parameters
        ----------
        other : Value or float
            Value to multiply (automatically wrapped if float)
            
        Returns
        -------
        Value
            Result of multiplication
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, other: Union[int, float]) -> 'Value':
        """
        Power operation: self ** other (other must be a constant)
        
        Gradient: d(a^n)/da = n * a^(n-1)
        
        Parameters
        ----------
        other : int or float
            Exponent (must be a constant, not a Value)
            
        Returns
        -------
        Value
            Result of exponentiation
            
        Raises
        ------
        AssertionError
            If other is not an int or float
        """
        assert isinstance(other, (int, float)), \
            "Only supports int/float powers for now"
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward
        return out

    def __neg__(self) -> 'Value':
        """
        Negation operation: -self
        
        Returns
        -------
        Value
            Negated value
        """
        return self * -1

    def __sub__(self, other: Union['Value', float]) -> 'Value':
        """
        Subtraction operation: self - other
        
        Implemented as self + (-other)
        
        Parameters
        ----------
        other : Value or float
            Value to subtract
            
        Returns
        -------
        Value
            Result of subtraction
        """
        return self + (-other)

    def __truediv__(self, other: Union['Value', float]) -> 'Value':
        """
        Division operation: self / other
        
        Implemented as self * (other ** -1)
        Gradient: d(a/b)/da = 1/b, d(a/b)/db = -a/b^2
        
        Parameters
        ----------
        other : Value or float
            Divisor
            
        Returns
        -------
        Value
            Result of division
        """
        return self * (other ** -1)

    def relu(self) -> 'Value':
        """
        ReLU activation: max(0, self)
        
        Gradient: d(ReLU(x))/dx = 1 if x > 0, else 0
        
        Returns
        -------
        Value
            Result of ReLU activation
        """
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    def tanh(self) -> 'Value':
        """
        Hyperbolic tangent activation: tanh(self)
        
        tanh(x) = (e^x - e^-x) / (e^x + e^-x)
        Gradient: d(tanh(x))/dx = 1 - tanh(x)^2
        
        Returns
        -------
        Value
            Result of tanh activation
        """
        x = self.data
        t = (np.exp(2 * x) - 1) / (np.exp(2 * x) + 1)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward
        return out

    def sigmoid(self) -> 'Value':
        """
        Sigmoid activation: 1 / (1 + e^-self)
        
        Gradient: d(sigmoid(x))/dx = sigmoid(x) * (1 - sigmoid(x))
        
        Returns
        -------
        Value
            Result of sigmoid activation
        """
        s = 1 / (1 + np.exp(-self.data))
        out = Value(s, (self,), 'sigmoid')

        def _backward():
            self.grad += s * (1 - s) * out.grad
        out._backward = _backward
        return out

    def exp(self) -> 'Value':
        """
        Exponential function: e^self
        
        Gradient: d(e^x)/dx = e^x
        
        Returns
        -------
        Value
            Result of exponential
        """
        out = Value(np.exp(self.data), (self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def log(self) -> 'Value':
        """
        Natural logarithm: log(self)
        
        Gradient: d(log(x))/dx = 1/x
        
        Returns
        -------
        Value
            Result of natural logarithm
            
        Raises
        ------
        ValueError
            If self.data <= 0 (log undefined for non-positive values)
        """
        if self.data <= 0:
            raise ValueError(f"log undefined for non-positive values (got {self.data})")
        out = Value(np.log(self.data), (self,), 'log')

        def _backward():
            self.grad += (1 / self.data) * out.grad
        out._backward = _backward
        return out

    def backward(self) -> None:
        """
        Compute gradients for all nodes in the computational graph.
        
        This method implements reverse-mode automatic differentiation
        (backpropagation). It:
        1. Builds a topological ordering of the computation graph
        2. Initializes this node's gradient to 1.0
        3. Propagates gradients backward through the graph
        
        After calling backward(), all Value nodes that contributed to
        this value will have their .grad attribute set to the derivative
        of this value with respect to them.
        
        Note: Gradients accumulate! Call zero_grad() before backward()
        if you want to reset gradients.
        
        Examples
        --------
        >>> x = Value(2.0)
        >>> y = Value(3.0)
        >>> z = x * y + x ** 2
        >>> z.backward()
        >>> print(x.grad)  # dz/dx = y + 2*x = 7
        7.0
        >>> print(y.grad)  # dz/dy = x = 2
        2.0
        """
        # Build topological ordering
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # Backward pass
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()

    def zero_grad(self) -> None:
        """
        Reset gradient to zero.
        
        Useful when reusing the same computational graph for multiple
        backward passes, as gradients accumulate by default.
        """
        self.grad = 0.0

    # Reverse operations (for when Value is on the right side)
    def __radd__(self, other: Union['Value', float]) -> 'Value':
        """Right addition: other + self"""
        return self + other

    def __rsub__(self, other: Union['Value', float]) -> 'Value':
        """Right subtraction: other - self"""
        other = other if isinstance(other, Value) else Value(other)
        return other - self

    def __rmul__(self, other: Union['Value', float]) -> 'Value':
        """Right multiplication: other * self"""
        return self * other

    def __rtruediv__(self, other: Union['Value', float]) -> 'Value':
        """Right division: other / self"""
        other = other if isinstance(other, Value) else Value(other)
        return other / self

    def __repr__(self) -> str:
        """String representation of Value"""
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    def __str__(self) -> str:
        """User-friendly string representation"""
        if self._op:
            return f"Value({self.data:.4f}) from {self._op}"
        return f"Value({self.data:.4f})"
