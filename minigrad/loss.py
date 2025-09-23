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
    loss = -sum([p.log() if t == 1 else 0 for p, t in zip(y_pred, y_target)])

    return loss