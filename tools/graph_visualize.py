from graphviz import Digraph

def trace(root):
  nodes, edges = set(), set() 
  def build(i):
    if i not in nodes:
      nodes.add(i)
      for child in i._prev:
        edges.add((child, i))
        build(child)
  build(root)
  return nodes, edges

def draw_dot(root):
  dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})
  
  nodes, edges = trace(root)
  for n in nodes:
    uid = str(id(n))
    dot.node(name = uid, label="{ %s | data %.4f | grad %.4f}" % (n.label , n.data, n.grad), shape='record')
    if n._op:
      # creating node/edge for an operation
      dot.node(name = uid + n._op, label=n._op)
      dot.edge(uid + n._op, uid)

  for n1, n2 in edges:
    dot.edge(str(id(n1)), str(id(n2)) + n2._op)

  return dot