import math

class Value:
  def __init__(self, data, _children=(), _op='', label=''):
    self.data = data
    self.grad = 0.0
    self._backward = lambda: None
    self._prev = _children
    self._op = _op
    self.label = label

  def __repr__(self):
    return f"Value(data={self.data})"
  
  def __add__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data + other.data, (self, other), '+')

    def _backward():
      self.grad += 1.0 * out.grad
      other.grad += 1.0 * out.grad
    out._backward = _backward

    return out
  
  def __radd__(self, other):
    return self + other
  
  def __neg__(self, ):
    return self * -1
  
  def __sub__(self, other):
    return self + (-1*other)
  
  def __rsub__(self, other):
    return self - other
  
  def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data * other.data, (self, other), '*')

    def _backward():
      self.grad += other.data * out.grad
      other.grad += self.data * out.grad
    out._backward = _backward

    return out
  
  def __rmul__(self, other):
    return self * other
  
  def __truediv__(self, other):
    return self * other**-1
  
  def __rtruediv__(self, other):
    return other * self**-1
  
  def __pow__(self, other):
    assert isinstance(other, (int, float)), "support for int/float only"
    out = Value(self.data ** other, (self, ), f'**{other}')

    def _backward():
      self.grad += other * (self.data ** (other - 1)) * out.grad
    out._backward = _backward

    return out
  
  def sigmoid(self):
    x = self.data
    s = 1 / (1 + math.exp(-x))
    out = Value(s, (self, ), 'sigmoid')

    def _backward():
      self.grad += (s * (1 - s)) * out.grad
    out._backward = _backward

    return out

  def tanh(self):
    x = self.data
    t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
    out = Value(t, (self, ), 'tanh')

    def _backward():
      self.grad += (1 - t**2) * out.grad
    out._backward = _backward

    return out
       
  def relu(self):
    out = Value(0 if self.data < 0 else self.data, (self, ), 'ReLU')
    
    def _backward():
      self.grad += (self.data > 0) * out.grad
    out._backward = _backward

    return out
  
  def leaky_relu(self):
    out = Value(0.01*self.data if self.data < 0 else self.data, (self, ), 'LReLU')

    def _backward():
      if self.data < 0:
        self.grad += 0.01 * out.grad
      else:
        self.grad += 1.0 * out.grad
    out._backward = _backward

    return out
  
  def linear(self):
    out = Value(self.data, (self, ), 'linear')

    def _backward():
      self.grad += 1.0 * out.grad
    out._backward = _backward

    return out 

  def exp(self):
    x = self.data
    out = Value(math.exp(x), (self, ), 'exp')

    def _backward():
      self.grad += out.data * out.grad
    out._backward = _backward

    return out
  
  def log(self):
    x = self.data
    out = Value(math.log(x), (self, ), 'log')

    def _backward():
      self.grad += (1 / (x * math.log(10))) * out.grad
    out._backward = _backward

    return out
  
  def softmax_cross_entropy(self, logits, targets):

    exps = [logit.exp() for logit in logits]
    s = sum(exps)
    softmax = [e / s for e in exps]

    loss_value = 0.0
    for sft, y in zip(softmax, targets):
      if y == 1:
        loss_value = -math.log(sft.data)
        break
    loss = Value(loss_value, tuple(logits), 'softmax_cross_entropy')

    def _backward():
      for logit, y, sft in zip(logits, targets, softmax):
        logit.grad += (sft.data - y) * logit.grad
    loss._backward = _backward

    return loss

  def backward(self):
    
    # topological sort
    topo = []
    visited = set()
    def build_topo(v):
      if v not in visited:
        visited.add(v)
        for child in v._prev:
          if isinstance(child, list):
            for c in child:
              build_topo(c)
          else:
            build_topo(child)
        topo.append(v)
    build_topo(self)

    self.grad = 1.0
    for node in reversed(topo):
      node._backward()