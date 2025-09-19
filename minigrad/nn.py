import random
from .engine import Value

class Neuron:
  def __init__(self, nin, activation='linear'):
    self.w = [Value(random.uniform(-1,1), label='w') for _ in range(nin)]
    self.b = Value(random.uniform(-1,1), label='b')
    self.activation = activation
    self.z = None
    self.out = None

  def __call__(self, x):
    # z = w * x + b
    self.z = sum([wi * xi for wi, xi in zip(self.w, x)], start=self.b)
    self.z.label = 'pre-activation'
    if self.activation == 'sigmoid':
      self.out = self.z.sigmoid()
    elif self.activation == 'tanh':
      self.out = self.z.tanh()
    elif self.activation == 'relu':
      self.out = self.z.relu()
    elif self.activation == 'leaky_rely':
      self.out = self.z.leaky_relu()
    else:
      self.out = self.z.linear()
    return self.out
  
  def __repr__(self):
    return f"Neuron({len(self.w)})"
  
  def parameters(self):
    return self.w + [self.b]
  
  def print_neuron(self, i, forward=False):
    if forward:
      print(f"\033[32mNeuron({i+1}) -> z={(self.z.data if self.z is not None else 0.0):.4f} -> {self.activation} -> out={(self.out.data if self.out is not None else 0.0):.4f}\033[0m")
    else:
      print(f"\033[32mNeuron({i+1}) -> \033[0m")
    params = self.parameters()
    for k, parameter in enumerate(params):
      print(f"  w[{k}] = {parameter.data:.4f}" if k != (len(params)-1) else f"  b = {parameter.data:.4f}")
    return
  
class Layer:
  def __init__(self, nin, nout, activation):
    self.neurons = [Neuron(nin, activation) for _ in range(nout)]
    self.nin = nin
    self.nout = nout
    self.activation = activation

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
  
  def print_layer(self, i, forward=False):
    print(f"\033[35m{i+1}.Layer ({self.nin}, {self.nout}){f" -> {self.activation}" if self.activation else ''}:\033[0m")
    for j, neuron in enumerate(self.neurons):
      neuron.print_neuron(j, forward)
    return

class MLP:
  def __init__(self, nin, nouts, activations=None):
    self.nin = nin
    self.nouts = nouts
    self.activations = activations
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
  
  def print_nn(self, forward=False):
    if self.activations and (len(self.nouts) == len(self.activations)):
      print(f"\033[36mMLP architecture: inputs({self.nin}) -> {' -> '.join([f'Layer({m}, {n})' for m, n in zip(self.nouts, self.activations)])}\033[0m")
    else:
      print(f"\033[36mMLP architecture: inputs({self.nin}) -> {' -> '.join([f'Layer({m}, linear)' for m in self.nouts])}\033[0m")

    for i, layer in enumerate(self.layers):
      layer.print_layer(i, forward)
    
    return None
  
  def zero_grad(self):
    for p in self.parameters():
      p.grad = 0.0