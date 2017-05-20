# What's this going to tell me?
This master cheatsheet will tell you...

1. some different statistical problems (regression, classification, etc.)
2. the techniques used to model solutions to those problems
3. the assumptions those techniques require
4. the metrics that can be used to measure the effectiveness of the resulting models
5. how to use those techniques in [R and/or Python](http://mathesaurus.sourceforge.net/r-numpy.html)

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
* R:
* Python:

## Neighbor Regression
* R: `knn.reg()` from FNN
* Python: `KNeighborsRegressor()` and `RadiusNeighborsRegressor()` from sklearn.neighbors

---

# Classification
## Logistic Regression
* R: `glm(family = binomial())` from the mlogit, or `mlogit()` if the output consists of multiple classes
* Python: `linear_model.LogisticRegression().fit()` from sklearn

## Decision Trees
* R:
* Python:

## Linear Discriminant Analysis and Quadratic Discriminant Analysis
* R: `lda(y ~ x)` and `qda(y ~ x)` from MASS
* Python: `LinearDiscriminantAnalysis.fit(x,y).predict(x)` and `QuadraticDiscriminantAnalysis.fit(x,y).predict(x)` from sklearn.discriminant_analysis

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

# Checking Assumptions
