[Source](https://numerics.mathdotnet.com/distance.html)

* Numerical Data
    * Difference-based
        * Manhattan Distance = L-1 Norm = Sum-of-Absolute-Distances
            * `Sum(abs(x_i - y_i))`
        * Euclidean Distance = L-2 Norm = Square-Root-of-Sum-of-Squared-Differences
            * `SquareRoot(Sum((x_i - y_i)^2))`
            * Root-Mean-Square-Distance
                * `SquareRoot(Sum((x_i - y_i)^2)/N)`
        * Chebyshev Distance = L-infinity Norm
            * `max_i(abs(x_i - y_i))`
    * Similarity-based
        * Cosine Similarity
            * `x.y / |x||y|`
            * represents the angular distance of two vectors (via dot product, ignoring magnitude)
        * Pearson's Correlation Coefficient
            * `cov(x, y) / sd(x) sd(y)`
            * measures linear correlation between x and y
            * ranges from -1 (total negative correlation) to +1 (total positive correlation)
* Non-Numerical Data
    * Hamming Distance
        * for vectors - the fewer elements are different, the more similar the vectors are
        * `count_i(x_i != y_x)`
    * Jaccard Similarity
        * for sets - the closer the size of the intersection is to the size of the union, the more similar the sets are
        * `size(intersection(x,y)) / size(union(x,y))`