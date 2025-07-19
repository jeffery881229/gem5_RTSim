import pydot

# 创建一个简单的图形
graph = pydot.Dot(graph_type='graph')

# 添加节点和边
node_a = pydot.Node("A")
node_b = pydot.Node("B")
graph.add_node(node_a)
graph.add_node(node_b)
graph.add_edge(pydot.Edge(node_a, node_b))

# 保存图形为文件
graph.write_png('example_graph.png')
