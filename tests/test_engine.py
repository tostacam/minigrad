import torch
import minigrad as mg
from minigrad import Value

# ------------------------------
def print_error(t, color='red'):
  if color == "red":
    print(f"\033[31m{t}\033[0m") 
  elif color == "cyan":
    print(f"\033[36m{t}\033[0m")
# ------------------------------  

def test_basic_ops():
  
  # testing -> add, radd, neg, sub, rsub, mul, rmul, truediv, rtruediv
  x = Value(3.0)
  z = (x + 2) + (-1.5 + x)
  q = (z - x) -(2 - x) + (z - 0.5)
  v = (q * 2) + (1.5 * q)
  v = (v / x) + (v / 2.0) + (10 / x)
  y = v**2
  y.backward()
  x_mg, y_mg = x, y

  x = torch.Tensor([3.0]).double()
  x.requires_grad = True
  z = (x + 2) + (-1.5 + x)
  q = (z - x) -(2 - x) + (z - 0.5)
  v = (q * 2) + (1.5 * q)
  v = (v / x) + (v / 2.0) + (10 / x)
  y = v**2
  y.backward()
  x_pt, y_pt = x, y 

  assert y_mg.data == y_pt.data.item(), print_error(f"Output mistmatch: [minigrad: {y_mg.data}, pytorch: {y_pt.data.item()}]")
  assert x_mg.grad == x_pt.grad.item(), print_error(f"Gradient mism")

def test_activation_functions():
  pass

  # TODO: testing -> tanh, relu, 


test_basic_ops()
test_activation_functions()