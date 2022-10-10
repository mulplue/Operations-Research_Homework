import numpy as np
from queue import PriorityQueue


class Graph:
    def __init__(self, n):
        self.v = n
        self.edges = [[-1 for i in range(n)] for j in range(n)]
        self.visited = []

    ## 有向图
    def add_edge(self, u, v, weight):
        self.edges[u][v] = weight


def dijkstra(graph, s):
        distance = [np.inf for i in range(graph.v)]
        distance[s] = 0         # 起始点离自己距离为0

        to_visit = PriorityQueue()
        to_visit.put((0,s))     # 访问起始点

        while not to_visit.empty():
            (dist, current_v) = to_visit.get()          # 取出最小的边
            graph.visited.append(current_v)             # 确认ok的点

            for neighbor in range(graph.v):
                if graph.edges[current_v][neighbor] != -1 and neighbor not in graph.visited:
                    d = graph.edges[current_v][neighbor]
                    old_dis = distance[neighbor]
                    new_dis = distance[current_v] + d
                    if new_dis < old_dis:
                        to_visit.put((new_dis, neighbor))
                        distance[neighbor] = new_dis
        return distance


if __name__ == '__main__':
    ## Init
    start_v = 1
    n = 6
    G = Graph(n+1)
    G.add_edge(1,2,1)
    G.add_edge(1,3,2)
    G.add_edge(2,3,3)
    G.add_edge(2,4,3)
    G.add_edge(2,6,7)
    G.add_edge(3,4,2)
    G.add_edge(3,5,2)
    G.add_edge(4,6,3)
    G.add_edge(5,6,6)

    ## Dijkstra
    distance = dijkstra(G, start_v)[1:]
    print(distance)

