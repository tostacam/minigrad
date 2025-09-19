import torch
import torch.nn as nn
from minigrad.engine import Value
from minigrad.nn import MLP, Layer, Neuron

def test_activation_functions():

    tol = 1e-6

    # Initialize inputs
    x_val = 0.5
    x = Value(x_val)
    x_torch = torch.tensor([x_val], dtype=torch.double, requires_grad=True)

    # Linear activation (identity)
    y_lin = x.linear()
    y_lin_torch = x_torch
    y_lin.backward()
    y_lin_torch.backward()
    assert abs(y_lin.data - y_lin_torch.data.item()) < tol, f"Linear error: [minigrad: {y_lin.data}, pytorch: {y_lin_torch.data.item()}]"
    assert abs(x.grad - x_torch.grad.item()) < tol, f"Linear error: [minigrad: {x.grad}, pytorch: {x_torch.grad.item()}]"
    x.grad = 0.0
    x_torch.grad.zero_()

    # Sigmoid activation
    y_sig = x.sigmoid()
    y_sig_torch = torch.sigmoid(x_torch)
    y_sig.backward()
    y_sig_torch.backward()
    assert abs(y_sig.data - y_sig_torch.data.item()) < tol, f"Sigmoid error: [minigrad: {y_sig.data}, pytorch: {y_sig_torch.data.item()}]"
    assert abs(x.grad - x_torch.grad.item()) < tol, f"Sigmoid error: [minigrad: {x.grad}, pytorch: {x_torch.grad.item()}]"
    x.grad = 0.0
    x_torch.grad.zero_()

    # Tanh activation
    y_tanh = x.tanh()
    y_tanh_torch = torch.tanh(x_torch)
    y_tanh.backward()
    y_tanh_torch.backward()
    assert abs(y_tanh.data - y_tanh_torch.data.item()) < tol, f"Tanh error: [minigrad: {y_tanh.data}, pytorch: {y_tanh_torch.data.item()}]"
    assert abs(x.grad - x_torch.grad.item()) < tol, f"Tanh error: [minigrad: {x.grad}, pytorch: {x_torch.grad.item()}]"
    x.grad = 0.0
    x_torch.grad.zero_()

    # ReLU activation
    y_relu = x.relu()
    y_relu_torch = torch.relu(x_torch)
    y_relu.backward()
    y_relu_torch.backward()
    assert abs(y_relu.data - y_relu_torch.data.item()) < tol, f"ReLU error: [minigrad: {y_relu.data}, pytorch: {y_relu_torch.data.item()}]"
    assert abs(x.grad - x_torch.grad.item()) < tol, f"ReLU error: [minigrad: {x.grad}, pytorch: {x_torch.grad.item()}]"
    x.grad = 0.0
    x_torch.grad.zero_()

    # Leaky ReLU activation
    y_lrelu = x.leaky_relu()
    y_lrelu_torch = torch.nn.functional.leaky_relu(x_torch)
    y_lrelu.backward()
    y_lrelu_torch.backward()
    assert abs(y_lrelu.data - y_lrelu_torch.data.item()) < tol, f"Leaky ReLU error: [minigrad: {y_lrelu.data}, pytorch: {y_lrelu_torch.data.item()}]"
    assert abs(x.grad - x_torch.grad.item()) < tol, f"Leaky ReLU error: [minigrad: {x.grad}, pytorch: {x_torch.grad.item()}]"
    x.grad = 0.0
    x_torch.grad.zero_()

def test_intermediate_gradients():

    tol = 1e-6

    # Build computation graph with intermediate nodes
    a = Value(1.5)
    b = Value(-2.0)
    c = a * b
    d = c.relu()
    e = d + a
    f = e.sigmoid()
    f.backward()

    a_torch = torch.tensor([1.5], dtype=torch.double, requires_grad=True)
    b_torch = torch.tensor([-2.0], dtype=torch.double, requires_grad=True)
    c_torch = a_torch * b_torch
    d_torch = torch.relu(c_torch)
    e_torch = d_torch + a_torch
    f_torch = torch.sigmoid(e_torch)
    f_torch.backward()

    # Compare forward results
    assert abs(c.data - c_torch.data.item()) < tol, f"forward error: [minigrad {c.data}, pytorch, {c_torch.data.item()}]"
    assert abs(d.data - d_torch.data.item()) < tol, f"forward error: [minigrad {d.data}, pytorch, {d_torch.data.item()}]"
    assert abs(e.data - e_torch.data.item()) < tol, f"forward error: [minigrad {e.data}, pytorch, {e_torch.data.item()}]"
    assert abs(f.data - f_torch.data.item()) < tol, f"forward error: [minigrad {f.data}, pytorch, {f_torch.data.item()}]"

    # Compare gradients of inputs and intermediate nodes
    assert abs(a.grad - a_torch.grad.item()) < tol, f"gradient error: [minigrad {a.data}, pytorch, {a_torch.data.item()}]"
    assert abs(b.grad - b_torch.grad.item()) < tol, f"gradient error: [minigrad {b.data}, pytorch, {b_torch.data.item()}]"

def test_mlp_against_pytorch():
    tol = 1e-5

    # dataset
    xs = [[0.5, -1.0], [1.0, 2.0]]
    ys = [[1.0], [0.0]]

    # minigrad
    mg_mlp = MLP(2, [3, 2, 1], activations=['tanh','relu','sigmoid'])

    # pytorch
    torch_mlp = nn.Sequential(
        nn.Linear(2, 3),
        nn.Tanh(),
        nn.Linear(3, 2),
        nn.ReLU(),
        nn.Linear(2, 1),
        nn.Sigmoid()
    )
    torch_mlp.double()

    # matching weights on pytorch
    with torch.no_grad():
        for mg_layer, torch_layer in zip(mg_mlp.layers, [layer for layer in torch_mlp if isinstance(layer, nn.Linear)]):
            if isinstance(torch_layer, nn.Linear):
                # weights
                torch_layer.weight.copy_(torch.tensor([[w.data for w in neuron.w] for neuron in mg_layer.neurons], dtype=torch.double))
                # biases
                torch_layer.bias.copy_(torch.tensor([neuron.b.data for neuron in mg_layer.neurons], dtype=torch.double))

    # forward minigrad
    mg_outputs = []
    for x in xs:
        x_vals = [Value(v) for v in x]
        out = mg_mlp(x_vals)
        mg_outputs.append(out)

    # forward pytorch
    x_tensor = torch.tensor(xs, dtype=torch.double, requires_grad=True)
    torch_outputs = torch_mlp(x_tensor)

    # loss
    mg_loss = Value(0.0)
    for out, y in zip(mg_outputs, ys):
        diff = out - y[0]
        mg_loss += diff * diff
    mg_loss /= len(xs)

    torch_y = torch.tensor(ys, dtype=torch.double)
    criterion = nn.MSELoss()
    torch_loss = criterion(torch_outputs, torch_y)

    # backward
    mg_loss.backward()
    torch_loss.backward()

    # comparison -> outputs
    for mg_out, torch_out in zip(mg_outputs, torch_outputs.detach().numpy()):
        assert abs(mg_out.data - torch_out[0]) < tol, f"Output mismatch: [minigrad {mg_out.data}, pytorch {torch_out[0]}]"

    # comparison -> gradients
    first_layer = mg_mlp.layers[0]
    torch_first_linear = torch_mlp[0]

    for i, neuron in enumerate(first_layer.neurons):
        for j, w in enumerate(neuron.w):
            torch_grad = torch_first_linear.weight.grad[i, j].item()
            mg_grad = w.grad
            assert abs(mg_grad - torch_grad) < tol, f"Weight gradient mismatch at layer 0 neuron {i} weight {j}: minigrad {mg_grad}, pytorch {torch_grad}"
        torch_bias_grad = torch_first_linear.bias.grad[i].item()
        mg_bias_grad = neuron.b.grad
        assert abs(mg_bias_grad - torch_bias_grad) < tol, f"Bias gradient mismatch at layer 0 neuron {i}: minigrad {mg_bias_grad}, pytorch {torch_bias_grad}"

def test_softmax_against_pytorch():
    tol = 1e-6

    # inputs -> logits
    logits_data = [[1.0, 2.0, 3.0], [0.5, 1.5, -1.0]]
    logits = [[Value(v) for v in row] for row in logits_data]

    # forward minigrad
    mg_softmax_outputs = []
    for row in logits:
        exps = [x.exp() for x in row]
        sum_exp = Value(0.0)
        for e in exps:
            sum_exp += e
        softmax_row = [e / sum_exp for e in exps]
        mg_softmax_outputs.append(softmax_row)

    # forward pytorch
    logits_tensor = torch.tensor(logits_data, dtype=torch.double, requires_grad=True)
    torch_softmax_outputs = nn.functional.softmax(logits_tensor, dim=1)

    # comparison -> outputs
    for i in range(len(logits_data)):
        for j in range(len(logits_data[0])):
            mg_val = mg_softmax_outputs[i][j].data
            torch_val = torch_softmax_outputs[i, j].item()
            assert abs(mg_val - torch_val) < tol, f"Softmax output mismatch at ({i},{j}): minigrad {mg_val}, pytorch {torch_val}"

    # backward
    mg_loss = Value(0.0)
    for row in mg_softmax_outputs:
        for val in row:
            mg_loss += val
    mg_loss.backward()

    torch_loss = torch_softmax_outputs.sum()
    torch_loss.backward()

    # comparison -> gradients
    for i in range(len(logits_data)):
        for j in range(len(logits_data[0])):
            mg_grad = logits[i][j].grad
            torch_grad = logits_tensor.grad[i, j].item()
            assert abs(mg_grad - torch_grad) < tol, f"Softmax gradient mismatch at ({i},{j}): minigrad {mg_grad}, pytorch {torch_grad}"

def test_mlp_softmax_against_pytorch():
    tol = 1e-5

    # dataset
    xs = [[0.5, -1.0], [1.0, 2.0]]
    ys = [[1, 0, 0], [0, 1, 0]]

    # minigrad MLP
    mg_mlp = MLP(2, [2, 3], activations=['relu', 'softmax'])  # last layer softmax applied directly

    # pytorch MLP
    torch_mlp = nn.Sequential(
        nn.Linear(2, 2),
        nn.ReLU(),
        nn.Linear(2, 3),
    )
    torch_mlp.double()

    # matching weights on pytorch
    with torch.no_grad():
        for mg_layer, torch_layer in zip(mg_mlp.layers, [layer for layer in torch_mlp if isinstance(layer, nn.Linear)]):
            torch_layer.weight.copy_(torch.tensor([[w.data for w in neuron.w] for neuron in mg_layer.neurons], dtype=torch.double))
            torch_layer.bias.copy_(torch.tensor([neuron.b.data for neuron in mg_layer.neurons], dtype=torch.double))

    # forward minigrad
    mg_outputs = []
    for x in xs:
        x_vals = [Value(v) for v in x]
        out = mg_mlp(x_vals)
        mg_outputs.append(out)

    # forward pytorch
    x_tensor = torch.tensor(xs, dtype=torch.double, requires_grad=True)
    torch_logits = torch_mlp(x_tensor)
    torch_softmax = nn.functional.softmax(torch_logits, dim=1)

    # compute cross-entropy loss minigrad
    mg_loss = Value(0.0)
    for out, y in zip(mg_outputs, ys):
        mg_loss += -sum([y_logit.log() if ygt == 1 else 0 for y_logit, ygt in zip(out, y)])
    mg_loss /= len(mg_outputs)

    # compute cross-entropy loss pytorch
    y_tensor = torch.tensor(ys, dtype=torch.double)
    torch_loss = -(y_tensor * torch_softmax.log()).sum(dim=1).mean()

    # backward
    mg_loss.backward()
    torch_loss.backward()

    # compare outputs
    for mg_out, torch_out in zip(mg_outputs, torch_softmax.detach().numpy()):
        for mg_val, torch_val in zip(mg_out, torch_out):
            assert abs(mg_val.data - torch_val) < tol, f"Softmax output mismatch: minigrad {mg_val.data}, pytorch {torch_val}"

    # compare gradients of first layer weights and biases
    first_layer = mg_mlp.layers[0]
    torch_first_linear = torch_mlp[0]

    for i, neuron in enumerate(first_layer.neurons):
        for j, w in enumerate(neuron.w):
            torch_grad = torch_first_linear.weight.grad[i, j].item()
            mg_grad = w.grad
            assert abs(mg_grad - torch_grad) < tol, f"Weight gradient mismatch at layer 0 neuron {i} weight {j}: minigrad {mg_grad}, pytorch {torch_grad}"
        torch_bias_grad = torch_first_linear.bias.grad[i].item()
        mg_bias_grad = neuron.b.grad
        assert abs(mg_bias_grad - torch_bias_grad) < tol, f"Bias gradient mismatch at layer 0 neuron {i}: minigrad {mg_bias_grad}, pytorch {torch_bias_grad}"

test_activation_functions()
test_intermediate_gradients()
test_mlp_against_pytorch()
test_softmax_against_pytorch()
test_mlp_softmax_against_pytorch()