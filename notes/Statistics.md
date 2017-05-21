# [Linear Discriminant Analysis (LDA)](http://sebastianraschka.com/Articles/2014_python_lda.html)
* dimensionality-reduction-via projection technique, similar to PCA, commonly used in pre-processing
* looks at classes, and reduce the number of features by creating axes in order to minimize the ratio `(within-group variance)/(between-group variance)`
	* we want groups with elements similar to other elements in the same group, and dissimilar to elements outside of the group
	* these axes are also called "latent variables" or "discriminants". These are the linear combinations of variables that distinguish classes most strongly.
* differences from PCA
	* supervised (class-based), unlike PCA (concerned with data only, not external classes)
	* finding axes which maximize separation between classes (unlike PCA, which is concerned with finding axes which maximize variance of the data)
* you could essentially use this as an alternative or supplement to feature selection techniques such as stepwise regression
* [can also be used as a classification technique](http://people.revoledu.com/kardi/tutorial/LDA/LDA.html#LDA)
	* assumptions
		* data is normally distributed
		* each class has identical variance-covariance matrices for the input variables
			* [if the matrices are substantially different, observations will be skewed towards the class with greater variance](https://stats.stackexchange.com/questions/71489/three-versions-of-discriminant-analysis-differences-and-how-to-use-them)
	* [to predict which class a new point belongs to, it approximates Bayes' Classifier, the the classfier with teh lowest possible eror rate](https://rpubs.com/ryankelly/LDA-QDA)
* `lda()` from MASS (has the same syntax as `lm()`)
	* returns...
		* the percentage of each group is in the data
		* the mean of each variable in each group
		* the coefficients of the old variables used to create the linear discriminant

## Quadratic Discriminant Analysis
* can learn quadratic boundaries, and is therefore more flexible than LDA
* allows for heterogeneity of classes' covariance matrices
	* estimates a separate covariance matrix for each class - adds computattion time
* `qda()` from MASS

---

# [Decision Trees](http://scikit-learn.org/stable/modules/tree.html)
* greedy classification technique that "splits" the data into a tree structure to locally maximize information gain (usually measured by a reduction in entropy, so that the leaves will eventually be high in purity)
	* many variations exist - some use Gini Impurity or Variance Reduction instead of Information Gain
* can also be used as a regressor (outputs will be in discrete, rather than continuous values)
* `rpart(y ~ x, method)` from rpart
	* where `method` is `"class"` for classification, or `"anova"` for regression

---

# [Nearest Neighbor](http://scikit-learn.org/stable/modules/neighbors.html)
* `neighbors.KNeighborsClassifier(n_neighbors).fit(x, y)` from sklearn
* neighbors may be weighted on their proximity to the queried point

## Distance Calculations
* Brute Force
	* compute distance between every pair of elements - slow, but no data structure overhead
	* O[DN]
* K-D Tree
	* heuristic to speed up distance calculations - if node A is close to node B, and node B is far from node C, then node A is far from node C.
	* O[Dlog(n)] for small n (&lt;20), O[DN] for large n (>20)
* Ball Tree
	* another heuristic - more expensive than K-D for low dimensions (&lt;20) but less expensive for high dimensions (>20)
		* divides data into hyperspheres. By knowing which hyperspheres A and B belong to, we can set an upper- and lower-bound on the distance between A and B
	* O[Dlog(n)]
* Which to use?
	* Brute force is fastest for small samples (N&lt;20) since it doesn't have to construct a data structure
	* Brute force is not affected by # of neighbors k (the others have to make additional queries, since each distance is not computed explicitly)
	* Generally...
		* Brute force is best if k is more than half of N
		* K-D is best if k is less than half of N and # dimensions is less than 20
		* Ball is best if k is less than half of N and # dimensions is more than 20
