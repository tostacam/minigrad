import matplotlib.pyplot as plt
import numpy as np

from minigrad.engine import Value
from minigrad.nn import MLP, Layer, Neuron

x_train = [[0,0],[0,1],[1,0],[1,1]]
y_train = [0,1,1,0]

# NN definition
model = MLP(2, [4, 1], activations=['linear', 'linear'])
print("number of parameters: ", len(model.parameters()))

# hyperparameters
steps = 100
learning_rate = 0.0025
loss_plot = [[],[]]

for k in range(steps):
  # forward pass
  v_values = [[Value(float(v)) for v in arr] for arr in x_train]
  ypred = [model(v) for v in v_values]
  loss = sum([(yout - ygt)**2 for ygt, yout in zip(y_train, ypred)])

  # backward pass
  model.zero_grad()
  loss.backward()

  # update weights
  for p in model.parameters():
    p.data += -learning_rate * p.grad

  loss_plot[0].extend([k])
  loss_plot[1].extend([loss.data])

plt.plot(loss_plot[0],loss_plot[1])
plt.xlabel('number of steps (k)')
plt.ylabel('loss')
plt.show()