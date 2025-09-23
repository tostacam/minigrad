import torch
import torch.nn as nn
from minigrad import Value, Neuron, Layer, MLP

# ------------------------------
def print_error(t):
  return f"\033[31m{t}\033[0m"
# ------------------------------

def test_neuron():
  tol = 1e-6

  # testing -> z = w * x + b, w[0].grad
  xs = [1.0, -1.5, 2.0, 0.5]

  # minigrad neuron
  mg_neuron = Neuron(4)

  # pytorch neuron + matching params
  pt_input = torch.tensor(xs, dtype=torch.double, requires_grad=True).unsqueeze(0)
  pt_neuron = torch.nn.Linear(in_features=4, out_features=1, dtype=torch.double)
  with torch.no_grad():
      pt_neuron.weight.copy_(torch.tensor([w.data for w in mg_neuron.w], dtype=torch.double))
      pt_neuron.bias.copy_(torch.tensor([mg_neuron.b.data], dtype=torch.double))

  # forward
  mg_pred = mg_neuron(xs)
  pt_pred = pt_neuron(pt_input)

  # backward
  mg_pred.backward()
  pt_pred.backward()

  assert abs(mg_pred.data - pt_pred.data.item()) < tol, print_error(f"Output mismatch: [minigrad: {mg_pred.data}, pytorch: {pt_pred.data.item()}]")
  assert abs(mg_neuron.w[0].grad - pt_neuron.weight.grad[0,0].item()) < tol, print_error(f"Gradient mismatch: [minigrad: {mg_neuron.w[0].grad}, pytorch: {pt_neuron.weight.grad[0,0].item()}]")

def test_layer():
  tol = 1e-6

  # testing -> z = W * x + b, neuron[0].w[0].grad
  xs = [2.0, -0.5, 3.0]

  # minigrad layer
  mg_layer = Layer(3, 4)

  # pytorch layer + matching params
  pt_input = torch.tensor(xs, dtype=torch.double, requires_grad=True).unsqueeze(0)
  pt_layer = torch.nn.Linear(in_features=3, out_features=4, dtype=torch.double)
  with torch.no_grad():
      pt_layer.weight.copy_(torch.tensor([[w.data for w in neuron.w] for neuron in mg_layer.neurons], dtype=torch.double))
      pt_layer.bias.copy_(torch.tensor([neuron.b.data for neuron in mg_layer.neurons], dtype=torch.double))
  
  # forward
  mg_sum = Value(0.0)
  mg_forward = [v for v in mg_layer([Value(x) for x in xs])]
  mg_pred = mg_sum.sum(mg_forward)
  pt_pred = pt_layer(pt_input)

  # backward
  mg_pred.backward()
  pt_pred.sum().backward()

  assert all(abs(mg_forward[i].data - pt_pred[0, i].item()) < tol for i in range(4)), print_error(f"Output mismatch: [minigrad: {mg_sum}, pytorch: {pt_pred}]")
  assert abs(mg_layer.neurons[0].w[0].grad - pt_layer.weight.grad[0,0].item()) < tol, print_error(f"Gradient mismatch: [minigrad: {mg_layer.neurons[0].w[0].grad}, pytorch: {pt_layer.weight.grad[0,0].item()}]")

def test_mlp():
  tol = 1e-6

  # testing -> z = W * x + b, layer[0].neuron[0].w[0].grad
  xs = [1.0, -2.5, 3.0]

  # minigrad MLP
  mg_mlp = MLP(3, [4, 4, 1], ["tanh", "relu", "sigmoid"])

  # pytorch MLP
  pt_mlp = nn.Sequential(
    nn.Linear(3, 4),
    nn.Tanh(),
    nn.Linear(4, 4),
    nn.ReLU(),
    nn.Linear(4, 1),
    nn.Sigmoid()
  )
  pt_mlp.double()

  # matching weights
  with torch.no_grad():
      for i, mg_layer in enumerate(mg_mlp.layers):
          pt_linear = pt_mlp[2*i]
          pt_linear.weight.copy_(torch.tensor([[w.data for w in neuron.w] for neuron in mg_layer.neurons], dtype=torch.double))
          pt_linear.bias.copy_(torch.tensor([neuron.b.data for neuron in mg_layer.neurons], dtype=torch.double))

  # forward
  mg_pred = mg_mlp(xs)
  pt_input = torch.tensor(xs, dtype=torch.double, requires_grad=True).unsqueeze(0)
  pt_pred = pt_mlp(pt_input)

  # backward
  mg_pred.backward()
  pt_pred.backward()

  assert abs(mg_pred.data - pt_pred.item()) < tol, print_error(f"Output mismatch: [minigrad: {mg_pred.data}, pytorch: {pt_pred.item()}]")
  assert abs(mg_mlp.layers[0].neurons[0].w[0].grad - pt_mlp[0].weight.grad[0,0].item()) < tol, print_error(f"Gradient mismatch: [minigrad: {mg_mlp.layers[0].neurons[0].w[0].grad}, pytorch: {pt_mlp[0].weight.grad[0,0].item()}]")

test_neuron()
test_layer()
test_mlp()