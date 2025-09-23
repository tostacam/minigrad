import torch
import torch.nn as nn
import minigrad as mg

# ------------------------------
def print_error(t):
  return f"\033[31m{t}\033[0m"
# ------------------------------

def test_MSE():
  tol = 1e-6

  # testing -> MSELoss()
  xs = [1.0, 2.0]
  ys = [3.0]

  # minigrad MLP
  mg_mlp = mg.MLP(2, [3, 3, 1], ["relu", "relu", "linear"])
  mg_loss_fn = mg.MSELoss()

  # pytorch MLP
  pt_mlp = nn.Sequential(
    nn.Linear(2, 3),
    nn.ReLU(),
    nn.Linear(3, 3),
    nn.ReLU(),
    nn.Linear(3, 1)
  ).double()
  torch_loss_fn = nn.MSELoss()

  # matching weights and biases
  with torch.no_grad():
    for i, mg_layer in enumerate(mg_mlp.layers):
        pt_linear = pt_mlp[2*i]
        pt_linear.weight.copy_(torch.tensor([[w.data for w in neuron.w] for neuron in mg_layer.neurons], dtype=torch.double))
        pt_linear.bias.copy_(torch.tensor([neuron.b.data for neuron in mg_layer.neurons], dtype=torch.double))

  # forward
  pt_input = torch.tensor(xs, dtype=torch.double).unsqueeze(0)
  pt_target = torch.tensor(ys, dtype=torch.double)
  
  mg_pred = [mg_mlp([mg.engine.Value(x) for x in xs])]
  pt_pred = pt_mlp(pt_input).squeeze(0)

  # loss
  mg_loss = mg_loss_fn(mg_pred, ys)
  pt_loss = torch_loss_fn(pt_pred, pt_target)

  # backward
  mg_loss.backward()
  pt_loss.backward()

  assert abs(mg_loss.data - pt_loss.item()) < tol, print_error(f"Loss mismatch: [minigrad: {mg_loss.data}, pytorch: {pt_loss.item()}]")


test_MSE()