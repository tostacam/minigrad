import torch
from minigrad import Neuron, Layer, MLP

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

  #outputs
  mg_pred = mg_neuron(xs)
  pt_pred = pt_neuron(pt_input)

  # backward
  mg_pred.backward()
  pt_pred.backward()

  assert abs(mg_pred.data - pt_pred.data.item()) < tol, print_error(f"Output mismatch: [minigrad: {mg_pred.data}, pytorch: {pt_pred.data.item()}]")
  assert abs(mg_neuron.w[0].grad - pt_neuron.weight.grad[0,0].item()) < tol, print_error(f"Gradient mismatch: [minigrad: {mg_neuron.w[0].grad}, pytorch: {pt_neuron.weight.grad[0,0].item()}]")
  
  mg_neuron.print_neuron()
  mg_neuron.print_neuron(type='grad')

test_neuron()