###In pre-flow push algorithm we have three main operations
## 1: Preprocess: In preprocess we:
#             (A) Set each vertex's height and flow to 0.
#             (B) Set the source vertex height to the total number of vertices in the graph.
#             (C) For each edge set flow to 0 at the beginning.
#             (D) Initially, flow and excess flow are equal to capacity for all vertices adjacent to source
## 2: Push(): operation is perform on a vertex having excess flow. We push the flow from a vertex to an adjacent vertex
#             with a lower height i.e if the vertex have excess flow and the adjacent node has a smaller height (in the
#             residual graph) then we push the minimum of excess flow and edge capacity to the adjacent vertex.
## 3: Relabel(): operation is used when there exist the excess flow in a vertex but we are unable to push that excess
#             flow because none of the adjacent vertex have lower height therefore relabel operation increase the height
#             of the vertex to perform push() operation. In order to Increases height we select the smallest height
#             adjacent (in the residual graph i.e. an adjacent to whom we can add flow) and adding one to it .

# We have use SGraph class to create input graphs.

# On the execution of the code we will get edge set, vertex set capacity assign to each edge, maximum flow in network
# and run time.

import sys
from Sgraph import Grid, CompleteGraph, RandomGraph, SGraph

import timeit, random


def getmaxflow(source, edgec, vert, Capacity):
    # initializing Edge dictionary for storing its keys and values
    Edge = dict()
    # initializing Vert dictionary for storing its keys and values
    Vert = dict()
    nflow = []
    nu = []

    print('EdgeSet = ', edgec)
    print('number of edges = ', len(edgec))
    print('VertexSet = ', vert)
    print('number of vertex = ', len(vert))
    print('Capacity = ', Capacity)
    print('assigned capacity = ', len(Capacity))

    # Intializing flow to 0 and capacity,u,v(from edgeset) according to the given data
    for id, indv_edge in enumerate(edgec, 1):
        # Extracting an edge from edgeset
        u, v = indv_edge
        Edge[id] = {'flow': 0, 'capacity': Capacity[id - 1], 'u': u, 'v': v}

    # Intializing height to 0 and excess flow to 0
    for id, vertices_data in enumerate(vert, 1):
        Vert[id] = {'height': 0, 'eflow': 0}

    def preprocess(source):
        # Initializing source height to the total no. of vertices
        Vert[source]['height'] = len(Vert)

        for id, info in Edge.items():
            # if edge is coming out of source
            if Edge[id]['u'] == source:
                # Saurate that edge by its capacity
                Edge[id]['flow'] = Edge[id]['capacity']
                # establishing the extra flow for the adjacent vertex
                Vert[Edge[id]['v']]['eflow'] += Edge[id]['flow']

                nflow.append(-Edge[id]['flow'])
                nu.append(Edge[id]['v'])
        # add an edge in the residual graph from v to s
        for i, j in zip(nflow, nu):
            Edge[len(Edge) + 1] = {'flow': i, 'capacity': 0, 'u': j, 'v': source}

    def checkoverflow(Vert):
        # check if we have an excess flow in intermediate vertex
        for id in range(2, len(Vert)):
            # check if currect vertex is greater than 0
            if Vert[id]['eflow'] > 0:
                return id
                # if no overflow vertex
        return -1

    def checkbackflow(id, flow):
        # Reverse flow to be update for the edge flow
        u = Edge[id]['v']
        v = Edge[id]['u']
        for id2, info in Edge.items():
            if Edge[id2]['u'] == u and Edge[id2]['v'] == v:
                Edge[id2]['flow'] -= flow
                return
        # addition of reverse edge in residual graph
        Edge[(len(Edge) + 1)] = {'flow': 0, 'capacity': flow, 'u': u, 'v': v}

    def push(id_v):
        # check edges adjacent to id_v on which flow can be pushed

        for id, info in Edge.items():
            # check to see whether the current edge u matches the specified overflow vertex id_v
            if Edge[id]['u'] == id_v:
                # There exist no push when flow equals to capacity
                if Edge[id]['flow'] == Edge[id]['capacity']:
                    continue

                # if height of a vertex is greater than the height of a vertex in which flow is to be pushed
                if Vert[id_v]['height'] > Vert[Edge[id]['v']]['height']:
                    # push minimum value between the excess flow of the vertex and residual capacity of that edge
                    flow = min(Edge[id]['capacity'] - Edge[id]['flow'], Vert[id_v]['eflow'])
                    # subract flow from the given excess flow vertex
                    Vert[id_v]['eflow'] -= flow
                    # increase excess flow to the adjacent vertex
                    Vert[Edge[id]['v']]['eflow'] += flow
                    # increase flow to the current edge flow
                    Edge[id]['flow'] += flow

                    checkbackflow(id, flow)
                    return True
        return False

    def relabel(id_v):
        # Determine the adjacent vertices minimum height
        min_height = sys.maxsize

        # identity the adjacent vertex with the least height
        for id, info in Edge.items():
            if Edge[id]['u'] == id_v:
                # if flow is equal to capacity then no relabeling operation can be done
                if Edge[id]['flow'] == Edge[id]['capacity']:
                    continue

                # update min_height
                if Vert[Edge[id]['v']]['height'] < min_height:
                    min_height = Vert[Edge[id]['v']]['height']
                    # Increasing id_v height
                    Vert[id_v]['height'] = min_height + 1

    preprocess(source)

    while (checkoverflow(Vert) != -1):
        id_v = checkoverflow(Vert)
        if not push(id_v):
            relabel(id_v)

    return Vert[len(Vert)]['eflow']


if __name__ == '__main__':
    # n=0
    # for j in range(5):

    # for running time
    start_time = timeit.default_timer()
    # n += 50
    ##GRAPH INPUT
    # check = Grid(20,20)

    # custom input graph
    # No of vertices=6
    edgec = [(1, 2), (1, 3), (3, 2), (2, 4), (5, 2), (3, 5), (5, 6), (4, 5), (4, 6)]
    Capacity = [7, 12, 4, 9, 3, 9, 6, 6, 15]
    vert = [1, 2, 3, 4, 5, 6]
    source = 1
    sink = 6
    # solution=23

    ## GET Edge sets from the graph and store into a variable
    # edgec=check.EdgeSet()

    ## GET Vertexes from the graph and store  into a variable
    # vert=check.VertexSet()

    # Assigning random capacity to each edge
    # Capacity = []
    # for i in range(len(edgec)):
    #    Capacity.append(random.randrange(1, 50))

    # source=1

    print("Maximum Flow = ", getmaxflow(source, edgec, vert, Capacity))
    print("Time duration", timeit.default_timer() - start_time)





