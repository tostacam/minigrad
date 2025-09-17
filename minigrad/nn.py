import random
from .engine import Value

class Neuron:
  def __init__(self, nin, activation='linear'):
    self.w = [Value(random.uniform(-1,1), label='w') for _ in range(nin)]
    self.b = Value(random.uniform(-1,1), label='b')
    self.activation = activation

  def __call__(self, x):
    # w * x + b
    act = sum([wi * xi for wi, xi in zip(self.w, x)], start=self.b)
    act.label = 'neuron(xi): ' + ' '.join(map(str,x))
    if self.activation == 'sigmoid':
      act.sigmoid()
    elif self.activation == 'tanh':
      act.tanh()
    elif self.activation == 'relu':
      act.relu()
    elif self.activation == 'leaky_rely':
      act.leaky_relu()
    else:
      act.linear()
    return act
  
  def __repr__(self):
    return f"Neuron({len(self.w)})"
  
  def parameters(self):
    return self.w + [self.b]
  
class Layer:
  def __init__(self, nin, nout, **kwargs):
    self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

  def __call__(self, x):
    outs = [n(x) for n in self.neurons]
    return outs[0] if len(outs) == 1 else outs
  
  def __repr__(self):
    return f"Layer of [{' '.join(str(neuron) for neuron in self.neurons)}]"
  
  def parameters(self):
    params = []
    for neuron in self.neurons:
      ps = neuron.parameters()
      params.extend(ps)
    return params

class MLP:
  def __init__(self, nin, nouts, activations=None):
    sz = [nin] + nouts

    # default nn -> linear for all layers
    if activations is None:
      activations = ["linear"] * len(nouts)

    self.layers = [Layer(sz[i], sz[i+1], activation=activations[i]) for i in range(len(nouts))]

  def __call__(self, x):
    for layer in self.layers:
      x = layer(x)
    return x
  
  def __repr__(self):
    return f"MLP of [{' '.join(str(layer) for layer in self.layers)}]"
  
  def parameters(self):
    params = []
    for layer in self.layers:
      p = layer.parameters()
      params.extend(p)
    return params
  
  def zero_grad(self):
    for p in self.parameters():
      p.grad = 0.0