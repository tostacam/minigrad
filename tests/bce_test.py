import minigrad as mg

# data set
xs = [1.0, 2.0]
ys = [1.0, 0.0]

# initializing nn and loss
mg_mlp = mg.MLP(2, [3, 3, 2], ['linear', 'linear', 'sigmoid'])
loss_fn = mg.BCELoss()

for k in range(20):
  # forward
  y_pred = mg_mlp([mg.engine.Value(x) for x in xs])
  loss = loss_fn(y_pred, ys)

  # backward
  mg_mlp.zero_grad()
  loss.backward()

  # udpate params
  for p in mg_mlp.parameters():
    p.data += - 0.01 * p.grad

  print(k, loss.data)

mg_mlp.print_nn()