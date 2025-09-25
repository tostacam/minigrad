from .engine import Value

class MSELoss():
  def __init__(self):
    return 

  def __call__(self, y_pred, y_target):
    loss = sum([(p - t)**2 for p, t in zip(y_pred, y_target)])

    return loss
  
class BCELoss():
  def __init__(self):
    return
  
  def __call__(self, y_pred, y_target):
    loss = -sum([t * p.log() + (1 - t) * ((1 - p).log()) for p, t in zip(y_pred, y_target)]) / len(y_target)

    return loss
  
class CELoss():
  def __init__(self):
    return

  def __call__(self, y_pred, y_target):
      if isinstance(y_target, int):
          return -y_pred[y_target] + (sum([p.exp() for p in y_pred])).log()

      losses = []
      for pred, target in zip(y_pred, y_target):
          if isinstance(pred, list):
              losses.append(-pred[target] + (sum([p.exp() for p in pred])).log())
          else:
              losses.append(-pred + pred.exp().log())
      return sum(losses) / len(losses)