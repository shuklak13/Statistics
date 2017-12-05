**Incomplete, plz don't judge**

* Good for
    * Chess: branching factor = 35
    * Maze: branching factor = 3

* Branching Factor:
    * BFS: O(d^b) for both time and space
        * b = avg branching factor
        * d = depth of goal node
    * DFS: O(d^b) for time, O(d) for space
        * So, DFS is typically better
    * Bidirectional Search
        * faster convergance

* Heuristics to pick a node
    * Maze: how far from the end are we? Ignore walls.
        * Favor nodes that are closer to the end via Euclidean distance, without actually seeing how long the true path is to the end.
    * Chess: what board position gives us the biggest point difference in our facor?

* [A-Star](https://en.wikipedia.org/wiki/A*_search_algorithm)
    * search all possible paths in weighted graph to find path w/ smallest cost to goal
    * minimize f(n) = g(n) + h(n)
        * g(n) = known cost of path from start node to node n
        * h(n) = heuristic (unknown) cost of path from node n to goal node
            * problem-specific
            * ex: Euclidean Distance
    * you can store all the f(n) values for all neighboring nodes in a priority queue
        * adding new neighbors to the priority queue until you reach the goal

* Adverserial Search
    * AlphaBeta
        * pruning algorithm to decrease # nodes evaluated during minimax algorithms