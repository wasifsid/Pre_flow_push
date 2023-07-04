# Pre Flow Push Algorithm

## Overview

The Pre Flow Push algorithm is a graph algorithm used to solve the maximum flow problem in a network. It is an improvement over the Ford-Fulkerson algorithm and provides better performance in practice. This README file provides an overview of the Pre Flow Push algorithm(alternatively, Push-relabel algorithm) introduced by Goldberg which is a slightly different approach to the computation of maximum flows. Push-relabel algorithms are really frequently the method of choice in practice since they are quick to solve the maximum-flow problem and also other flow problems, such as the minimum-cost flow problem. Push-relabel algorithms work on one vertex at a time instead of finding an augmented path in the entire residual network and for this reason, push-relabel algorithms do not maintain the flow-conservation constraints throughout their execution but they do maintain a preflow. For more theoretical background and concepts visit my project https://drive.google.com/file/d/1ObXMm5aAseCBmTIYlLDKGLgaVFm6JbgM/view?usp=sharing   

## Algorithm Description

The Pre Flow Push algorithm works by maintaining a "pre-flow" in the network. A pre-flow assigns flow values to edges in the network without satisfying the flow conservation constraint. It starts with an initial feasible pre-flow and gradually transforms it into a maximum flow by pushing excess flow from higher-level vertices to lower-level vertices.

The algorithm follows these steps:

1. **Initialization**: Assign a pre-flow with positive flow values to the edges of the network, satisfying capacity constraints. Set the flow value of the source node to the sum of outgoing edge capacities.

2. **Push operation**: Select a vertex with excess flow (the flow entering the vertex is greater than the flow leaving the vertex). Push the excess flow to its adjacent vertices along the edges that have residual capacity. This operation reduces the excess flow at the current vertex and increases the flow at the adjacent vertices.

3. **Relabel operation**: If there are no more vertices with excess flow, then relabel a vertex. Relabeling increases the height of a vertex, which allows more flow to be pushed towards it in the next push operation.

4. **Repeat**: Repeat steps 2 and 3 until all excess flows are eliminated. The algorithm terminates when there are no vertices with excess flow.

5. **Maximum flow**: The final flow values assigned to the edges represent the maximum flow from the source to the sink in the network.

## Implementation

The Pre Flow Push algorithm can be implemented using various data structures and programming languages. Here is a high-level description of the implementation:

1. Represent the network as a directed graph with vertices and edges. Each edge has a capacity and a flow value associated with it.

2. Initialize the graph with the required data structures, such as adjacency lists or matrices to represent the graph structure, capacities, and flow values.

3. Implement the push operation to find a vertex with excess flow and push the flow to its adjacent vertices. Update the flow values and residual capacities accordingly.

4. Implement the relabel operation to increase the height of a vertex. This operation ensures that the push operation can continue to make progress towards the sink.

5. Use a suitable algorithm to find the maximum flow by repeating the push and relabel operations until no more excess flows exist.

6. Return the flow values assigned to the edges, which represent the maximum flow from the source to the sink in the network.

## Conclusion

The Pre Flow Push algorithm is an efficient approach for solving the maximum flow problem in a network. It provides an improvement over the Ford-Fulkerson algorithm by maintaining a pre-flow and using push and relabel operations to reduce the excess flow and increase the flow towards the sink. By following the guidelines and steps mentioned in this README, you can implement and utilize the Pre Flow Push algorithm in
