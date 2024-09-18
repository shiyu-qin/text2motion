import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class graph_kit():

    def __init__(self,
                 max_hop=1,
                 dilation=1):
        self.max_hop  = max_hop
        self.dilation = dilation
        self.lvls     = 5  # 21-14-9-4-1 假设有4个层级，如有不同，可以调整
        self.As       = []
        self.hop_dis  = []

        self.get_edge()
        # print(self.map)
        for lvl in range(self.lvls):
            self.hop_dis.append(get_hop_distance(self.num_node, self.edge, lvl, max_hop=max_hop))
            self.get_adjacency(lvl)
        self.mapping = upsample_mapping(self.map, self.nodes, self.edge, self.lvls)[::-1]
        for x in enumerate(self.mapping):
            print(len(x),x)

    def __str__(self):
        return str(self.As)

    def get_edge(self):
        self.num_node = []
        self.nodes = []
        self.center = [0,0,0,0,0]  # 将中心节点设置为1号节点（索引0）
        self.nodes = []
        self.Gs = []
        remove = [2,2,1,1]
        # neighbor_base = [(1,0),(2,1),(3,2),(4,3),(5,3),(6,5),(7,6),(8,3),
        #                  (9,8),(10,9),(11,1),(12,11),(13,12),(14,13),(15,14),
        #                  (16,1),(17,16),(18,17),(19,18),(20,19)]
        neighbor_base = [(0,2),(0,7),(0,15),(1,3),(1,4),(2,3),(1,17),(1,9),
                         (17,14),(14,18),(9,6),(6,10),(15,16),(16,13),(13,19),
                         (19,20),(7,8),(8,5),(5,12),(12,11)]
        
        neighbor_link = [(i, j) for (i, j) in neighbor_base]

        nodes = np.array([i for i in range(21)])
        G = nx.Graph()
        G.add_nodes_from(nodes)
        G.add_edges_from(neighbor_link)
        G = nx.convert_node_labels_to_integers(G, first_label=0)

        self_link = [(int(i), int(i)) for i in G]
# #创建了一个节点的映射，其中每个节点会被映射成一个 [i, x] 的数组，i 是节点的索引，x 是图中对应的节点
        self.map = [np.array([[i, x] for i,x in enumerate(G)])]
        self.edge = [np.array(G.edges)]
        self.nodes.append(nodes)
        self.num_node.append(len(G))
        self.Gs.append(G.copy())
         
        for lvl in range(self.lvls-1):
            start = remove[lvl]
            stay  = []
            re = []
            for i in G:
                if len(G.edges(i)) == start and i not in stay:
                    lost = []
                    for j,k in G.edges(i):
                        stay.append(k)
                        lost.append(k)
                    recon = [(l,m) for l in lost for m in lost if l!=m]
                    G.add_edges_from(recon)
                    re.append(i)
            G.remove_nodes_from(re)
            # plt.clf()
            # nx.draw(G,with_labels = True)
            # plt.show()
            map_i = np.array([[i, x] for i,x in enumerate(G)])  # 记录图索引
            self.map.append(map_i)
            # mapping = {}  # 修改映射标签
            # for i, x in enumerate(G): 
            #     mapping[int(x)] = i
            #     if int(x)==self.center[-1]:
            #         self.center.append(i)

            # G = nx.relabel_nodes(G, mapping)  # 更改标签
            G = nx.convert_node_labels_to_integers(G, first_label=0)

            nodes = np.array([i for i in range(len(G))])
            self.nodes.append(nodes)
            G_l = np.array(G.edges)
            self.edge.append(G_l)
            self.num_node.append(len(G))
            self.Gs.append(G.copy())
        # for lvl in range(self.lvls-1):
            # print(self.num_node)
            # print(self.edge)
            # print(self.map)
            # print(self.nodes)
        assert len(self.num_node) == self.lvls
        assert len(self.nodes)    == self.lvls
        assert len(self.edge)     == self.lvls
        assert len(self.center)   == self.lvls
        assert len(self.map)      == self.lvls
        
    def get_adjacency(self, lvl):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node[lvl], self.num_node[lvl]))
        for hop in valid_hop:
            adjacency[self.hop_dis[lvl] == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)

        A = []
        for hop in valid_hop:
            a_root = np.zeros((self.num_node[lvl], self.num_node[lvl]))
            a_close = np.zeros((self.num_node[lvl], self.num_node[lvl]))
            a_further = np.zeros((self.num_node[lvl], self.num_node[lvl]))
            for i in range(self.num_node[lvl]):
                for j in range(self.num_node[lvl]):
                    if self.hop_dis[lvl][j, i] == hop:
                        if self.hop_dis[lvl][j, self.center[lvl]] == self.hop_dis[lvl][i, self.center[lvl]]:
                            a_root[j, i] = normalize_adjacency[j, i]
                        elif self.hop_dis[lvl][j, self.center[lvl]] > self.hop_dis[lvl][i, self.center[lvl]]:
                            a_close[j, i] = normalize_adjacency[j, i]
                        else:
                            a_further[j, i] = normalize_adjacency[j, i]
            if hop == 0:
                A.append(a_root)
            else:
                A.append(a_root + a_close)
                A.append(a_further)
        A = np.stack(A)
        self.As.append(A)

def get_hop_distance(num_node, edge, lvl, max_hop=1):
    A = np.zeros((num_node[lvl], num_node[lvl]))
    for i, j in edge[lvl]:
        A[i,i] = 1
        A[j,j] = 1
        A[j, i] = 1
        A[i, j] = 1

    hop_dis = np.zeros((num_node[lvl], num_node[lvl])) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis

def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    return AD

def upsample_mapping(mapping, nodes, edges, lvls):
    all_hoods = []
    i = lvls - 1
    while i > 0:
        n = i - 1

        neighbors = []
        #下一层的节点列表
        for node in nodes[n]:
            if node not in mapping[i][:,1]:
                hood = []
                for cmap in mapping[i]:
                    hood.append(cmap[0]) if ([node, cmap[1]] in edges[n].tolist()) or ([cmap[1], node] in edges[n].tolist()) else None
                if len(hood)>0: hood.insert(0, node)
                if len(hood)>0: neighbors.append(np.array(hood)) 
        all_hoods.append(neighbors)
        i -= 1
    # 用于保存所有层次图中的“邻居集”（hoods）信息。
    return all_hoods
        

if __name__ == '__main__':
    graph_kit()


