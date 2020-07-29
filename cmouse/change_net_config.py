import network
import mousenet
import networkx as nx
import numpy as np

def change_net_config(net):

    new_net = network.Network()
    new_net.area_channels = net.area_channels
    new_net.area_size = net.area_size


    max_kernel_size = 9
    
    G, _ = net.make_graph()
    Gtop = nx.topological_sort(G)
    root = next(Gtop) # get root of graph
    edge_bfs = [e for e in nx.edge_bfs(G, root)] 

    for e in edge_bfs:
        layer = net.find_conv_source_target(e[0],e[1])
        params = layer.params

        if params.kernel_size > max_kernel_size:

            # params.out_channels = int(np.round((params.kernel_size / max_kernel_size) * params.out_channels))
            params.kernel_size = max_kernel_size
            params.gsw = 4
            if params.stride == 1:
                params.padding = 4
            elif params.stride == 2:
                params.padding = (3,4,3,4)

            layer.params = params

        new_net.layers.append(layer)
        print(e[0]+e[1],'kernel_size',params.kernel_size,'out_channels',params.out_channels, 'padding',params.padding)

    return new_net

if __name__ == "__main__":
    
    net = network.load_network_from_pickle('../example/network_(3,64,64).pkl')
    new_net = change_net_config(net)
    G, _ = new_net.make_graph()
    Gtop = nx.topological_sort(G)
    root = next(Gtop)