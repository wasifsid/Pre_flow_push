"""
A Class for Simple Undirected Graphs

SGraph stores a graph as a dictionary of neighborhood sets of vertices.
It provides basic graph operations and operators for
(1) deletion of edges, vertices, or vertex sets (-),
(2) insertion of edges (+),
(3) contraction of edges (/)
(4) merging a vertex set (/)
(5) disjoint union of graphs (|),
(6) Cartesian product of graphs (*).
In addition. the modul contains the following functions:
- building the complement of a graph: Complement
- LineGraph, InducedSubgraph
- generating special graphs: Path, Cycle, CompleteGraph, CompleteBipartite, Hypercube, Grid, Torus
- generating random graphs
- testing properties: connected, is_bipartite, is_path, is_complete, two_edege_connected
- minimum and maximum vertex degree: delta and Delta
- AdjacencyMatrix
- finding all components and articulations of a graph
- distance-related measures: Distance, DistanceVector, Eccentricity,
  Diameter, Radius, Center.
"""
from sympy import Poly, factorial, combsimp,expand
from sympy.abc import x

inf = 1000000000  # defines integer infinity,


# should be larger than the number of vertices of any graph

def Complement(G):
    V = G.VertexSet()
    E = G.EdgeSet()
    return SGraph(V, set([(u, v) for u in V for v in V
                          if u < v and not (u, v) in E]))


class SGraph:
    """A class for simple undirected graphs."""

    def __init__(self, V=set(), E=set()):
        self.G = dict((v, set()) for v in V)
        for e in E:
            u = e[0]
            w = e[1]
            self.G[u].add(w)
            self.G[w].add(u)

    def VertexSet(self):
        return set(self.G.keys())

    def EdgeSet(self):
        return set([(u, v) for u in self.G.keys() for v in self.G[u] if u < v])

    def Order(self):
        return len(self.VertexSet())

    def Size(self):
        return len(self.EdgeSet())

    def Neighbors(self, v):
        return self.G[v]

    def Degree(self, v):
        return len(self.G[v])

    def copy(self):
        return SGraph(self.VertexSet(), self.EdgeSet())

    def adjacent(self, u, v):
        return u in self.Neighbors(v)

    def InsertEdge(self, u, v):
        if not self.adjacent(u, v):
            self.G[u].add(v)
            self.G[v].add(u)

    def InsertVertex(self, v):
        if not v in self.VertexSet():
            self.G[v] = set()

    def __add__(self, x):
        H = self.copy()
        H.InsertEdge(x[0], x[1])
        return H

    def DeleteEdge(self, e):
        u = e[0]
        v = e[1]
        if u in self.Neighbors(v):
            self.G[u] -= {v}
            self.G[v] -= {u}

    def DeleteVertex(self, v):
        if v in self.VertexSet():
            for w in self.G[v]:
                self.G[w] -= {v}
            del self.G[v]

    def DeleteVertexSet(self, X):
        for v in X:
            self.DeleteVertex(v)

    def __sub__(self, x):
        H = self.copy()
        if x in self.EdgeSet():
            H.DeleteEdge(x)
        elif x in self.VertexSet():
            H.DeleteVertex(x)
        elif x <= self.VertexSet():
            H.DeleteVertexSet(x)
        return H

    def ContractEdge(self, e):
        u = e[0]
        v = e[1]
        self.DeleteEdge(e)
        for w in self.Neighbors(v):
            self.G[w] = (self.G[w] - {v}) | {u}
        self.G[u] |= self.G[v]
        self.DeleteVertex(v)

    def Merge(self, X):
        v = X.pop()
        self.G[v] -= X
        for w in X:
            for t in self.Neighbors(w):
                self.G[t] = (self.G[t] - {w}) | {v}
            self.G[v] |= self.G[w]
            self.DeleteVertex(w)

    def __truediv__(self, x):
        H = self.copy()
        if x in self.EdgeSet():
            H.ContractEdge(x)
        elif x <= self.VertexSet():
            H.Merge(x)
        return H

    def Product(self, H):
        V = [(u, v) for u in self.VertexSet() for v in H.VertexSet()]
        T = dict([(V[i], i + 1) for i in range(0, len(V))])
        E = set([(T[x], T[y]) for x in V for y in V
                 if x < y and ((x[0] == y[0] and x[1] in H.Neighbors(y[1])) or
                               (x[0] in self.Neighbors(y[0]) and x[1] == y[1]))])
        return SGraph(set(range(1, len(V) + 1)), E)

    def __mul__(self, H):
        return self.Product(H)

    def Normalize(self, shift=0):
        V = list(self.VertexSet())
        T = dict([(V[i - 1], i + shift) for i in range(1, len(V) + 1)])
        self.G = dict([(T[v], set(map(lambda x: T[x], self.G[v]))) for v in V])

    def DisjointUnion(self, H):
        A = self.copy()
        A.Normalize()
        B = H.copy()
        B.Normalize(A.Order())
        return SGraph(A.VertexSet() | B.VertexSet(), A.EdgeSet() | B.EdgeSet())

    def __or__(self, H):
        return self.DisjointUnion(H)


def LineGraph(G):
    X = list(map(set, list(G.EdgeSet())))
    m = G.Size()
    H = SGraph(set(range(1, m + 1)))
    for i in range(0, m - 1):
        for j in range(i + 1, m):
            if X[i] & X[j] != set():
                H.InsertEdge(i + 1, j + 1)
    return H


def Join(G, H):
    A = G.copy()
    B = H.copy()
    A.Normalize()
    B.Normalize(G.Order())
    U = A.VertexSet()
    V = B.VertexSet()
    E = A.EdgeSet() | B.EdgeSet() | set([(u, v) for u in U for v in V])
    return SGraph(U | V, E)


def Path(n):
    return SGraph(set(range(1, n + 1)), set((u, u + 1) for u in range(1, n)))


def Cycle(n):
    return SGraph(set(range(1, n + 1)),
                  set((u, u + 1) for u in range(1, n)) | {(1, n)})


def EmptyGraph(n):
    return SGraph(set(range(1, n + 1)))


def CompleteGraph(n):
    return Complement(EmptyGraph(n))


def CompleteBipartite(p, q):
    return SGraph(set(range(1, p + q + 1)),
                  set((u, v) for u in range(1, p + 1) for v in range(p + 1, p + q + 1)))


def Hypercube(d):
    G = Path(2)
    for i in range(1, d):
        G = G * Path(2)
    return G


def Grid(k, l):
    return Path(k) * Path(l)


def Torus(k, l):
    return Cycle(k) * Cycle(l)


def connected2(G):  # Caution! Causes stack overflow for large graphs.
    """DFS connectivity test. Works well for small garphs."""
    marked = dict([(v, 0) for v in G.VertexSet()])
    n = 0

    def DFS(v):
        nonlocal n
        marked[v] = 1
        n += 1
        for w in G.Neighbors(v):
            if not marked[w]:
                DFS(w)

    DFS(next(iter(G.VertexSet())))
    return n == G.Order()


def connected(G):
    """Connectivity test using BFS."""
    u = next(iter(G.VertexSet()))
    X = {u}
    Y = G.Neighbors(u)
    k = 1
    while Y - X != set():
        for w in Y - X:
            k += 1
        Z = Y - X
        X = X | Y
        Y = Y.union(*[G.Neighbors(x) for x in Z])
    return k == G.Order()


def FindPath(G, u, v):
    """Returns a shortest path from u to v in G as a vertex sequence."""
    from queue import Queue
    d = {t: "none" for t in G.VertexSet()}
    d[u] = 0
    f = dict()
    f[u] = "start"
    q = Queue()
    q.put(u)
    while not q.empty():
        x = q.get()
        for y in G.Neighbors(x):
            if d[y] == "none":
                d[y] = d[x] + 1
                f[y] = x
                if y == v:
                    path = [v]
                    while f[path[-1]] != u:
                        path.append(f[path[-1]])
                    return [u] + list(reversed(path))
                q.put(y)


def ispath(G, u, v):
    """Returns true if there is a path from u to v in G."""
    from queue import Queue
    d = {t: "none" for t in G.VertexSet()}
    d[u] = 0
    q = Queue()
    q.put(u)
    while not q.empty():
        x = q.get()
        for y in G.Neighbors(x):
            if d[y] == "none":
                d[y] = d[x] + 1
                if y == v: return True
                q.put(y)
    return False


def Delta(G):
    """Maximum degree of G."""
    return max([G.Degree(v) for v in G.VertexSet()])


def delta(G):
    """Minimum degree of G."""
    return min([G.Degree(v) for v in G.VertexSet()])


def AdjacencyMatrix(G):
    return [[int(G.adjacent(u, v))
             for v in G.VertexSet()] for u in G.VertexSet()]


def DistanceVector(G, u):
    """Length of a shortest path in G from u to all other vertices."""
    X = {u}
    Y = G.Neighbors(u)
    d = dict([(w, 2 ** 32) for w in G.VertexSet()])  # 2**32=infinity (almost)
    k = 0
    d[u] = k
    while Y - X != set():
        k += 1
        for w in Y - X:
            d[w] = k
        Z = Y - X
        X = X | Y
        Y = Y.union(*[G.Neighbors(x) for x in Z])
    return d


def CountWalks(G, u, v, k):
    """The procedure counts walks of length k between u and v."""
    if u == v and k == 0:
        return 1
    elif u != v and k == 0:
        return 0
    else:
        s = 0
        for w in G.Neighbors(u):
            s += CountWalks(G, w, v, k - 1)
        return s


def Distance(G, u, v):
    return DistanceVector(G, u)[v]


def Eccentricity(G, v):
    return max(DistanceVector(G, v).values())


def Radius(G):
    return min([Eccentricity(G, v) for v in G.VertexSet()])


def Diameter(G):
    return max([Eccentricity(G, v) for v in G.VertexSet()])


def Center(G):
    r = Radius(G)
    return filter(lambda v: Eccentricity(G, v) == r, G.VertexSet())


def is_bipartite(G):
    if G.Order() == 0:
        return True
    else:
        v = next(iter(G.VertexSet()))
        d = DistanceVector(G, v)
        for e in G.EdgeSet():
            if d[e[0]] == d[e[1]]:
                return False
        return True


def RandomGraph(n, p):
    from random import random
    G = SGraph(set(range(1, n + 1)))
    for u in range(1, n):
        for v in range(u + 1, n + 1):
            if random() < p:
                G.InsertEdge(u, v)
    return G


def InducedSubgraph(G, X):
    H = SGraph(X)
    for e in G.EdgeSet():
        if set(e) <= X:
            H.InsertEdge(*e)
    return H


def Component(G, v):
    """Returns the component of G containing vertex v."""
    X = set()
    Y = {v}
    Z = {v}
    while Z != set():
        for w in Z:
            Y |= G.Neighbors(w)
            X |= {w}
        Z = Y - X
    return InducedSubgraph(G, Y)


def Components(G):
    C = []
    V = G.VertexSet()
    while V != set():
        v = V.pop()
        H = Component(G, v)
        C.append(H)
        V -= H.VertexSet()
    return C


def Articulations(G):
    low_point = dict()
    dfs_number = dict([(v, 0) for v in G.VertexSet()])
    father = dict([(v, 0) for v in G.VertexSet()])
    tree_edges = set()
    k = 0

    def DFS(v):
        nonlocal k, tree_edges, low_point, father
        k += 1
        dfs_number[v] = k
        low_point[v] = k
        for w in G.Neighbors(v):
            if dfs_number[w] == 0:
                father[w] = v
                tree_edges |= {(v, w)}
                DFS(w)
                low_point[v] = min(low_point[v], low_point[w])
            elif w != father[v]:
                low_point[v] = min(low_point[v], dfs_number[w])

    v = next(iter(G.VertexSet()))
    DFS(v)
    A = set()
    for e in tree_edges:
        if e[0] != v and low_point[e[1]] >= dfs_number[e[0]]:
            A |= {e[0]}
    L = set(filter(lambda x: v in x, tree_edges))
    if len(L) > 1:
        A |= {v}
    return A


def biconnected(G):
    return connected(G) and len(Articulations(G)) == 0


def two_edge_connected(G):
    if G.Order() == 2 or not connected(G) or delta(G) == 1:
        return False
    elif biconnected(G):
        return True
    else:
        A = list(Articulations(G))
        k = len(A)
        if k == 1:
            return True
        else:
            for i in range(0, k - 1):
                for j in range(i + 1, k):
                    if G.adjacent(A[i], A[j]):
                        return False
            return True


def is_complete(G):
    return G.Order() == 0 or G.Order() - 1 == delta(G)


def is_empty(G):
    return G.Size == 0 or len(G.EdgeSet()) == 0


def is_tree(G):
    return connected(G) and G.Size() == (G.Order() - 1)
