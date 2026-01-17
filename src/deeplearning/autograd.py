"""
Autograd Module
===============

This module implements automatic differentiation (autograd) for both scalar and
tensor values. It provides a computational graph that tracks operations and
computes gradients via reverse-mode automatic differentiation (backpropagation).

Classes:
--------
- Value: Wraps a scalar and builds a dynamic computational graph
- Tensor: Wraps a numpy array for vector/matrix operations with autograd

Supported Operations:
---------------------
Value (scalar):
- Arithmetic: +, -, *, /, **
- Activations: ReLU, tanh, sigmoid
- Mathematical: exp, log
- Unary: negation

Tensor (vector/matrix):
- Arithmetic: +, -, *, /, ** (element-wise with broadcasting)
- Matrix ops: matmul (@), transpose, reshape, sum
- Activations: relu, tanh, sigmoid, softmax
- Mathematical: exp, log
- Indexing and slicing

Example:
--------
>>> from deeplearning.autograd import Value, Tensor
>>> # Scalar example
>>> x = Value(2.0)
>>> y = Value(3.0)
>>> z = x * y + x ** 2
>>> z.backward()
>>> print(f"dz/dx = {x.grad}")  # Should be y + 2*x = 3 + 4 = 7
>>> print(f"dz/dy = {y.grad}")  # Should be x = 2

>>> # Tensor example
>>> X = Tensor([[1, 2], [3, 4]])
>>> W = Tensor([[0.5, -0.5], [-0.5, 0.5]])
>>> y = X @ W
>>> y.sum().backward()
>>> print(W.grad)  # Gradient for weight update
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


class Tensor:
    """
    Multi-dimensional array with automatic differentiation support.

    This class wraps numpy arrays and builds a computational graph for
    automatic gradient computation. It supports element-wise operations,
    matrix multiplication, and various activation functions.

    Parameters
    ----------
    data : array-like
        Input data (list, numpy array, or scalar)
    _children : Tuple[Tensor, ...], optional
        Parent nodes in the computation graph
    _op : str, optional
        Operation that created this tensor
    requires_grad : bool, optional
        Whether to track gradients for this tensor (default: True)

    Attributes
    ----------
    data : np.ndarray
        The underlying numpy array
    grad : np.ndarray or None
        Gradient of the loss with respect to this tensor
    shape : Tuple[int, ...]
        Shape of the tensor
    requires_grad : bool
        Whether gradients are tracked

    Examples
    --------
    Basic operations:
    >>> x = Tensor([[1, 2], [3, 4]])
    >>> y = Tensor([[5, 6], [7, 8]])
    >>> z = x + y
    >>> print(z.data)
    [[ 6  8]
     [10 12]]

    Matrix multiplication with gradients:
    >>> W = Tensor([[0.1, 0.2], [0.3, 0.4]])
    >>> x = Tensor([[1, 2]])
    >>> y = x @ W
    >>> y.sum().backward()
    >>> print(W.grad)  # Gradients for weight update
    """

    def __init__(
        self,
        data: Union[np.ndarray, List, float, int],
        _children: Tuple['Tensor', ...] = (),
        _op: str = '',
        requires_grad: bool = True
    ):
        if isinstance(data, np.ndarray):
            self.data = data.astype(np.float64)
        else:
            self.data = np.array(data, dtype=np.float64)

        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data) if requires_grad else None
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    @property
    def shape(self) -> Tuple[int, ...]:
        """Return the shape of the tensor."""
        return self.data.shape

    @property
    def ndim(self) -> int:
        """Return the number of dimensions."""
        return self.data.ndim

    @property
    def size(self) -> int:
        """Return the total number of elements."""
        return self.data.size

    @property
    def T(self) -> 'Tensor':
        """Return the transpose of the tensor."""
        return self.transpose()

    def __add__(self, other: Union['Tensor', np.ndarray, float, int]) -> 'Tensor':
        """
        Element-wise addition with broadcasting support.

        Gradient: d(a + b)/da = 1, d(a + b)/db = 1
        For broadcasting, gradients are summed along broadcast dimensions.
        """
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(self.data + other.data, (self, other), '+')

        def _backward():
            if self.requires_grad:
                grad = out.grad
                # Handle broadcasting: sum along axes that were broadcast
                if self.shape != out.shape:
                    grad = _unbroadcast(grad, self.shape)
                self.grad += grad
            if other.requires_grad:
                grad = out.grad
                if other.shape != out.shape:
                    grad = _unbroadcast(grad, other.shape)
                other.grad += grad
        out._backward = _backward
        return out

    def __mul__(self, other: Union['Tensor', np.ndarray, float, int]) -> 'Tensor':
        """
        Element-wise multiplication with broadcasting support.

        Gradient: d(a * b)/da = b, d(a * b)/db = a
        """
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(self.data * other.data, (self, other), '*')

        def _backward():
            if self.requires_grad:
                grad = other.data * out.grad
                if self.shape != out.shape:
                    grad = _unbroadcast(grad, self.shape)
                self.grad += grad
            if other.requires_grad:
                grad = self.data * out.grad
                if other.shape != out.shape:
                    grad = _unbroadcast(grad, other.shape)
                other.grad += grad
        out._backward = _backward
        return out

    def __pow__(self, other: Union[int, float]) -> 'Tensor':
        """
        Element-wise power operation.

        Gradient: d(a^n)/da = n * a^(n-1)
        """
        assert isinstance(other, (int, float)), "Only supports int/float powers"
        out = Tensor(self.data ** other, (self,), f'**{other}')

        def _backward():
            if self.requires_grad:
                self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward
        return out

    def __neg__(self) -> 'Tensor':
        """Negation: -self"""
        return self * -1

    def __sub__(self, other: Union['Tensor', np.ndarray, float, int]) -> 'Tensor':
        """Subtraction: self - other"""
        return self + (-other)

    def __truediv__(self, other: Union['Tensor', np.ndarray, float, int]) -> 'Tensor':
        """Division: self / other"""
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        return self * (other ** -1)

    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        """
        Matrix multiplication: self @ other

        For 2D tensors: standard matrix multiplication
        For higher dimensions: batch matrix multiplication

        Gradient: d(A @ B)/dA = grad @ B.T, d(A @ B)/dB = A.T @ grad
        """
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(self.data @ other.data, (self, other), '@')

        def _backward():
            if self.requires_grad:
                if self.ndim == 1 and other.ndim == 2:
                    # (n,) @ (n, m) -> (m,)
                    self.grad += out.grad @ other.data.T
                elif self.ndim == 2 and other.ndim == 1:
                    # (n, m) @ (m,) -> (n,)
                    self.grad += np.outer(out.grad, other.data)
                else:
                    self.grad += out.grad @ np.swapaxes(other.data, -2, -1)
            if other.requires_grad:
                if self.ndim == 1 and other.ndim == 2:
                    # (n,) @ (n, m) -> (m,)
                    other.grad += np.outer(self.data, out.grad)
                elif self.ndim == 2 and other.ndim == 1:
                    # (n, m) @ (m,) -> (n,)
                    other.grad += self.data.T @ out.grad
                else:
                    other.grad += np.swapaxes(self.data, -2, -1) @ out.grad
        out._backward = _backward
        return out

    def sum(self, axis: Union[int, Tuple[int, ...], None] = None, keepdims: bool = False) -> 'Tensor':
        """
        Sum elements along specified axis.

        Parameters
        ----------
        axis : int, tuple of ints, or None
            Axis or axes along which to sum. None sums all elements.
        keepdims : bool
            Whether to keep the reduced dimensions

        Returns
        -------
        Tensor
            Sum of elements
        """
        out_data = self.data.sum(axis=axis, keepdims=keepdims)
        out = Tensor(out_data, (self,), 'sum')

        def _backward():
            if self.requires_grad:
                grad = out.grad
                # Broadcast gradient back to original shape
                if axis is not None and not keepdims:
                    if isinstance(axis, int):
                        grad = np.expand_dims(grad, axis)
                    else:
                        for ax in sorted(axis):
                            grad = np.expand_dims(grad, ax)
                self.grad += np.broadcast_to(grad, self.shape)
        out._backward = _backward
        return out

    def mean(self, axis: Union[int, Tuple[int, ...], None] = None, keepdims: bool = False) -> 'Tensor':
        """
        Compute mean along specified axis.
        """
        n = self.data.size if axis is None else np.prod([self.shape[i] for i in (axis if isinstance(axis, tuple) else (axis,))])
        return self.sum(axis=axis, keepdims=keepdims) / n

    def transpose(self, *axes) -> 'Tensor':
        """
        Transpose the tensor.

        Parameters
        ----------
        axes : int, optional
            Permutation of axes. If not provided, reverses all axes.
        """
        if not axes:
            axes = None
        out = Tensor(np.transpose(self.data, axes), (self,), 'T')

        def _backward():
            if self.requires_grad:
                if axes is None:
                    self.grad += np.transpose(out.grad)
                else:
                    # Compute inverse permutation
                    inv_axes = np.argsort(axes)
                    self.grad += np.transpose(out.grad, inv_axes)
        out._backward = _backward
        return out

    def reshape(self, *shape) -> 'Tensor':
        """
        Reshape the tensor to a new shape.

        Parameters
        ----------
        shape : int or tuple of ints
            New shape (must have same total number of elements)
        """
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        out = Tensor(self.data.reshape(shape), (self,), 'reshape')

        def _backward():
            if self.requires_grad:
                self.grad += out.grad.reshape(self.shape)
        out._backward = _backward
        return out

    def flatten(self) -> 'Tensor':
        """Flatten tensor to 1D."""
        return self.reshape(-1)

    def relu(self) -> 'Tensor':
        """
        ReLU activation: max(0, x)

        Gradient: 1 if x > 0, else 0
        """
        out = Tensor(np.maximum(0, self.data), (self,), 'relu')

        def _backward():
            if self.requires_grad:
                self.grad += (self.data > 0) * out.grad
        out._backward = _backward
        return out

    def tanh(self) -> 'Tensor':
        """
        Hyperbolic tangent activation.

        Gradient: 1 - tanh(x)^2
        """
        t = np.tanh(self.data)
        out = Tensor(t, (self,), 'tanh')

        def _backward():
            if self.requires_grad:
                self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward
        return out

    def sigmoid(self) -> 'Tensor':
        """
        Sigmoid activation: 1 / (1 + exp(-x))

        Gradient: sigmoid(x) * (1 - sigmoid(x))
        """
        s = 1 / (1 + np.exp(-np.clip(self.data, -500, 500)))  # Clip for numerical stability
        out = Tensor(s, (self,), 'sigmoid')

        def _backward():
            if self.requires_grad:
                self.grad += s * (1 - s) * out.grad
        out._backward = _backward
        return out

    def softmax(self, axis: int = -1) -> 'Tensor':
        """
        Softmax activation along specified axis.

        softmax(x)_i = exp(x_i) / sum(exp(x_j))

        Parameters
        ----------
        axis : int
            Axis along which to compute softmax (default: -1)
        """
        # Subtract max for numerical stability
        shifted = self.data - np.max(self.data, axis=axis, keepdims=True)
        exp_vals = np.exp(shifted)
        s = exp_vals / np.sum(exp_vals, axis=axis, keepdims=True)
        out = Tensor(s, (self,), 'softmax')

        def _backward():
            if self.requires_grad:
                # Jacobian-vector product for softmax
                # For each sample: grad_input = s * (grad_output - sum(grad_output * s))
                sum_grad_s = np.sum(out.grad * s, axis=axis, keepdims=True)
                self.grad += s * (out.grad - sum_grad_s)
        out._backward = _backward
        return out

    def log(self) -> 'Tensor':
        """
        Natural logarithm.

        Gradient: 1/x
        """
        out = Tensor(np.log(np.clip(self.data, 1e-12, None)), (self,), 'log')

        def _backward():
            if self.requires_grad:
                self.grad += (1 / np.maximum(self.data, 1e-12)) * out.grad
        out._backward = _backward
        return out

    def exp(self) -> 'Tensor':
        """
        Exponential function.

        Gradient: exp(x)
        """
        out = Tensor(np.exp(np.clip(self.data, -500, 500)), (self,), 'exp')

        def _backward():
            if self.requires_grad:
                self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def __getitem__(self, idx) -> 'Tensor':
        """
        Indexing and slicing support.

        Parameters
        ----------
        idx : int, slice, tuple, or array
            Index or slice specification
        """
        out = Tensor(self.data[idx], (self,), 'getitem')

        def _backward():
            if self.requires_grad:
                grad = np.zeros_like(self.data)
                grad[idx] = out.grad
                self.grad += grad
        out._backward = _backward
        return out

    def backward(self) -> None:
        """
        Compute gradients via reverse-mode automatic differentiation.

        This initiates backpropagation from this tensor (typically a scalar loss).
        Builds topological ordering and propagates gradients backward.
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

        # Initialize gradient for the output
        self.grad = np.ones_like(self.data)

        # Backward pass
        for v in reversed(topo):
            v._backward()

    def zero_grad(self) -> None:
        """Reset gradient to zeros."""
        if self.requires_grad:
            self.grad = np.zeros_like(self.data)

    # Reverse operations
    def __radd__(self, other: Union['Tensor', np.ndarray, float, int]) -> 'Tensor':
        """Right addition."""
        return self + other

    def __rsub__(self, other: Union['Tensor', np.ndarray, float, int]) -> 'Tensor':
        """Right subtraction."""
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        return other - self

    def __rmul__(self, other: Union['Tensor', np.ndarray, float, int]) -> 'Tensor':
        """Right multiplication."""
        return self * other

    def __rtruediv__(self, other: Union['Tensor', np.ndarray, float, int]) -> 'Tensor':
        """Right division."""
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        return other / self

    def __rmatmul__(self, other: Union['Tensor', np.ndarray]) -> 'Tensor':
        """Right matrix multiplication."""
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        return other @ self

    def __repr__(self) -> str:
        """String representation."""
        return f"Tensor(shape={self.shape}, requires_grad={self.requires_grad})"

    def __str__(self) -> str:
        """User-friendly string representation."""
        return f"Tensor({self.data})"

    def __len__(self) -> int:
        """Return length of first dimension."""
        return len(self.data)

    def numpy(self) -> np.ndarray:
        """Return the underlying numpy array."""
        return self.data

    def item(self) -> float:
        """Return scalar value (only for single-element tensors)."""
        return self.data.item()

    def detach(self) -> 'Tensor':
        """Return a new tensor detached from the computation graph."""
        return Tensor(self.data.copy(), requires_grad=False)

    @staticmethod
    def zeros(shape: Tuple[int, ...], requires_grad: bool = True) -> 'Tensor':
        """Create a tensor of zeros."""
        return Tensor(np.zeros(shape), requires_grad=requires_grad)

    @staticmethod
    def ones(shape: Tuple[int, ...], requires_grad: bool = True) -> 'Tensor':
        """Create a tensor of ones."""
        return Tensor(np.ones(shape), requires_grad=requires_grad)

    @staticmethod
    def randn(shape: Tuple[int, ...], requires_grad: bool = True) -> 'Tensor':
        """Create a tensor with random values from standard normal distribution."""
        return Tensor(np.random.randn(*shape), requires_grad=requires_grad)

    @staticmethod
    def rand(shape: Tuple[int, ...], requires_grad: bool = True) -> 'Tensor':
        """Create a tensor with random values from uniform [0, 1)."""
        return Tensor(np.random.rand(*shape), requires_grad=requires_grad)


def _unbroadcast(grad: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
    """
    Sum gradient along broadcast dimensions to match target shape.

    When broadcasting occurs during forward pass (e.g., adding a (3,) tensor
    to a (2, 3) tensor), the gradient needs to be summed along the broadcast
    dimensions during backward pass.

    Parameters
    ----------
    grad : np.ndarray
        Gradient from output
    target_shape : tuple
        Shape to reduce gradient to

    Returns
    -------
    np.ndarray
        Gradient with correct shape
    """
    # Handle scalar case
    if target_shape == ():
        return np.sum(grad)

    # Add leading dimensions if needed
    while grad.ndim > len(target_shape):
        grad = grad.sum(axis=0)

    # Sum along dimensions that were broadcast (size 1 in target)
    for i, (grad_dim, target_dim) in enumerate(zip(grad.shape, target_shape)):
        if target_dim == 1 and grad_dim > 1:
            grad = grad.sum(axis=i, keepdims=True)

    return grad
