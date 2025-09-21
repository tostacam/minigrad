import torch
from minigrad import Value

# ------------------------------
def print_error(t):
  return f"\033[31m{t}\033[0m"
# ------------------------------  

def test_basic_ops():
  
  # testing -> add, radd, neg, sub, rsub, mul, rmul, truediv, rtruediv, pow
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

  assert y_mg.data == y_pt.data.item(), print_error(f"Output mismatch: [minigrad: {y_mg.data}, pytorch: {y_pt.data.item()}]")
  assert x_mg.grad == x_pt.grad.item(), print_error(f"Gradient mismatch: [minigrad: {x_mg.grad}, pytorch: {x_pt.grad.item()}]")

def test_math_ops():

  # testing -> exp, log
  x = Value(4.0)
  z = x.exp()
  v = z + 2
  y = v.log()
  y.backward()
  x_mg, y_mg = x, y

  x = torch.Tensor([4.0]).double()
  x.requires_grad = True
  z = x.exp()
  v = z + 2
  y = v.log()
  y.backward()
  x_pt, y_pt = x, y

  assert y_mg.data == y_pt.data.item(), print_error(f"Output mismatch: [minigrad: {y_mg.data}, pytorch: {y_pt.data.item()}]")
  assert x_mg.grad == x_pt.grad.item(), print_error(f"Gradient mismatch: [minigrad: {x_mg.grad}, pytorch: {x_pt.grad.item()}]")

def test_activation_functions():

  # testing -> sigmoid
  x = Value(0.5)
  y = x.sigmoid() #sigmoid
  y.backward()
  x_mg, y_mg = x, y

  x = torch.Tensor([0.5]).double()
  x.requires_grad = True
  y = x.sigmoid() #sigmoid
  y.backward()
  x_pt, y_pt = x, y

  assert y_mg.data == y_pt.data.item(), print_error(f"Output mismatch: [minigrad: {y_mg.data}, pytorch: {y_pt.data.item()}]")
  assert x_mg.grad == x_pt.grad.item(), print_error(f"Gradient mismatch: [minigrad: {x_mg.grad}, pytorch: {x_pt.grad.item()}]")

  # testing -> tanh
  x = Value(2.0)
  y = x.tanh() #tanh
  y.backward()
  x_mg, y_mg = x, y

  x = torch.Tensor([2.0]).double()
  x.requires_grad = True
  y = x.tanh() #tanh
  y.backward()
  x_pt, y_pt = x, y

  assert y_mg.data == y_pt.data.item(), print_error(f"Output mismatch: [minigrad: {y_mg.data}, pytorch: {y_pt.data.item()}]")
  assert x_mg.grad == x_pt.grad.item(), print_error(f"Gradient mismatch: [minigrad: {x_mg.grad}, pytorch: {x_pt.grad.item()}]")

  # testing -> relu
  x = Value(-4.0)
  y = x.relu() #relu
  y.backward()
  x_mg, y_mg = x, y

  x = torch.Tensor([-4.0]).double()
  x.requires_grad = True
  y = x.relu() #relu
  y.backward()
  x_pt, y_pt = x, y

  assert y_mg.data == y_pt.data.item(), print_error(f"Output mismatch: [minigrad: {y_mg.data}, pytorch: {y_pt.data.item()}]")
  assert x_mg.grad == x_pt.grad.item(), print_error(f"Gradient mismatch: [minigrad: {x_mg.grad}, pytorch: {x_pt.grad.item()}]")

  # testing -> leaky_relu
  x = Value(-2.5)
  y = x.leaky_relu() #leaky_relu
  y.backward()
  x_mg, y_mg = x, y

  x = torch.Tensor([-2.5]).double()
  x.requires_grad = True
  y = torch.nn.functional.leaky_relu(x, negative_slope=0.01) #leaky_relu
  y.backward()
  x_pt, y_pt = x, y

  assert y_mg.data == y_pt.data.item(), print_error(f"Output mismatch: [minigrad: {y_mg.data}, pytorch: {y_pt.data.item()}]")
  assert x_mg.grad == x_pt.grad.item(), print_error(f"Gradient mismatch: [minigrad: {x_mg.grad}, pytorch: {x_pt.grad.item()}]")

test_basic_ops()
test_math_ops()
test_activation_functions()