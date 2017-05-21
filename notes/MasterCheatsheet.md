# What's this going to tell me?
This master cheatsheet will tell you...

1. some different statistical problems (regression, classification, etc.)
2. the techniques used to model solutions for those problems
3. the assumptions those techniques require
4. the metrics that can be used to measure the effectiveness of the resulting models
5. the pros and cons of each technique
6. how to use those techniques in [R and/or Python](http://mathesaurus.sourceforge.net/r-numpy.html)

# Regression
## Linear Regression
### Ordinary Least-Squares Linear Regression
* R: `lm(y ~ x)`
* Python: `linear_model.LinearRegression().fit(x)` from sklearn

### Robust Linear Regression

### Polynomial Least-Squares Regression
* R: `lm(y ~ poly(x, 3))` to perform a regression of the form `y = ax^3 + bx^2 + cx + d`
* Python: `linear_model.LinearRegression.fit(PolynomialFeatures(degree=2).transform(X))` from sklearn and sklearn.preprocessing

## Decision Tree Regression
* R: `rpart(y ~ x, "anova")` from rpart
* Python: `tree.DecisionTreeRegressor().fit(x, y)` from sklearn

## Neighbor Regression
* R: `knn.reg()` from FNN
* Python: `KNeighborsRegressor()` and `RadiusNeighborsRegressor()` from sklearn.neighbors

## Accuracy Metrics

---

# Classification
## Logistic Regression
* Assumptions:
* Advantages:
	* relatively robust: no assumptions are made about the data's distribution
* R: `glm(family = binomial())` from the mlogit, or `mlogit()` if the output consists of multiple classes
* Python: `linear_model.LogisticRegression().fit()` from sklearn

## Decision Trees
* Advantages
	* extremely easy to interpret
	* requires little data preparation (no normalization, dummy vars, feature selection required)
	* few assumptions about the data
* Disadvantages
	* unstable - small variations produce a completely different decision tree
	* greedy - optimizes locally, not globally
	* very vulnerable to overfitting - make sure to use pruning or validation/ensemble methods to correct for this
	* poor at out-of-sample prediction
* R: `rpart(y ~ x, "class")` from rpart
* Python: `tree.DecisionTreeClassifier().fit(x, y)` from sklearn

## Linear Discriminant Analysis and Quadratic Discriminant Analysis
* Assumptions:
	* data is normally distributed
	* homoscedasticity (each class has identical variance-covariance matrices)
		* [if not, observations will be skewed towards the class with greater variance](https://stats.stackexchange.com/questions/71489/three-versions-of-discriminant-analysis-differences-and-how-to-use-them)
		* Quadratic Discriminant relaxes this assumption, so variance matrices may be heterogeneous
* [comparision of LDA to Logistic Regression](http://mrvar2.fdv.uni-lj.si/pub/mz/mz1.1/pohar.pdf)
	* note that these two methods perform very similarly for large samples
	* Advantages over Logistic Regression:
		* more stable for small, normally-distributed samples
		* performs better for multiclass problems
	* Disadvantages to Logistic Regression:
		* data must be normally distributed
		* LDA's assumptions are so rarely met
		* performs worse for 2-class problems
* comparision of QDA and LDA
	* When to use QDA over LDA
		* if the class variances are expected to be different
		* if the class boundaries are expected to be nonlinear
	* When to use LDA over QDA
		* if the class boundary is expected to be linear
		* if n is small (estimating covariance matrices leads to high bias on a small sample)
* [R:](https://rpubs.com/ryankelly/LDA-QDA) `lda(y ~ x)` and `qda(y ~ x)` from MASS
* [Python:](http://scikit-learn.org/stable/modules/lda_qda.html) `LinearDiscriminantAnalysis.fit(x,y).predict(x)` and `QuadraticDiscriminantAnalysis.fit(x,y).predict(x)` from sklearn.discriminant_analysis

## K-Nearest Neighbors
* R: `knn(training, testing, trainingY, k)` from class
* Python: `neighbors.KNeighborsClassifier(n_neighbors).fit(x, y)` from sklearn`
* Unsupervised, so no model needs to be created

## Accuracy Metrics
* accuracy = (# correct)/(# incorrect)
	* should be compared against the naive accuracy (blindly guessing the most common class each time)
* precision = (# true positives) / (# true positives + false positives)
* recall = (# true positives) / (# true positives + false negatives)
* F-score = 2 \* precision \* recall / (precision + recall)
* Area under the Receiver-Operating-Characteristic Curve
	* a curve measuring the growth of the true-positive-rate (y-axis) relative to the false-positive-rate (x-axis) as a model's threshold is reduced

---

# Checking for Assumptions and Correcting Data
* In general, it's almost always assumed that the quantity of data exceeds the quantity of features. Most methods will be ineffective if this is not true, and and some methods may have impossible computations.

## Normality

## Homogeneous Variance

## Independent of Observations

## Feature Selection

## Nonlinear relationships
