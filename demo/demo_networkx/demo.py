import networkx as nx

if __name__ == '__main__':
    # G1 = nx.cycle_graph(6)
    # for node in G1.nodes():
    #     print(node)
    # G2 = nx.wheel_graph(7)
    G1 = nx.Graph()
    G2 = nx.Graph()
    for idx in range(10):
        G1.add_node((idx, idx + 1))
        G2.add_node((idx * 2, idx * 2 + 1))
        # G1.add_edge((idx, idx + 1), (idx * 2, idx * 2 + 1))
    res = nx.optimize_graph_edit_distance(G1, G2)
    print(res)