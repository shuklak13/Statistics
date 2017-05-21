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

## SVM Regression
* [R: `svm(y ~ x)` from libsvm](https://cran.r-project.org/web/packages/e1071/vignettes/svmdoc.pdf)
	* Note that SVM regression and classification both use the same code! The function determines whether to perform regression or classification depending on whether y is continuous or categorical.
* [Python: `LinearSVR`, `SVR`, and `NuSVR` from sklearn](http://scikit-learn.org/stable/modules/svm.html)

## Accuracy Metrics
* R^2 (Coefficient of Determination)
	* [where R is the correlation between Y and Y_hat](https://stats.stackexchange.com/questions/134167/is-there-any-difference-between-r2-and-r2)
		* unlike r, which is the correlation between X and Y (equal for simple linear regression)
	* measures the proportion of y's variance explained by the model
	* note that this is NOT always equal to the square of the correlation of r
* Adjusted R^2
	* penalizes the model for adding unnecessary features
* AIC and BIC
	* similar to Adjusted R^2
	* unlike R^2, cannot be used as a standalone measure
* F-Ratio
	* ratio of variance the model can't explain over the variance the model can explain
* R: `summary()` and `anova()`
* Python: [scikit-learn](http://scikit-learn.org/stable/modules/model_evaluation.html)

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
* Assumptions
	* all parameters can be mapped onto a hyperspace with a meaningful distance function
* Advantages
	* Lazy-Learner - no model needs to be created, so no training time!
	* Nonparametric - no assumptions about underlying data distribution
* Disadvantages
	* because all the computations are shifted from the training phase to the testing phase, the testing phase is more computationally expensive.
	* all data points need to be stored, so high memory consumption
* R: `knn(training, testing, trainingY, k)` from class
* Python: `neighbors.KNeighborsClassifier(n_neighbors).fit(x, y)` from sklearn

## Support Vector Machine
* Assumptions
* Advantages
	* effective in high-dimensional data
	* works well with both linearly and nonlinearly separable data [(nonlinearly-separable data uses the kernel trick)](https://en.wikipedia.org/wiki/Support_vector_machine#Nonlinear_classification)
	* finds the global minimum
	* memory efficient - only a few points are used to create decision boundary
	* surprisingly effective when # dimensions > # observations
* Disadvantages
	* does not return a confidence value
	* difficult to interpret model
	* they can scare people
* [R: `svm(y ~ x)` from libsvm](https://cran.r-project.org/web/packages/e1071/vignettes/svmdoc.pdf)
* [Python: `LinearSVC`, `SVC`, and `NuSVC` from sklearn](http://scikit-learn.org/stable/modules/svm.html)

## Accuracy Metrics
* accuracy = (# correct)/(# incorrect)
	* should be compared against the naive accuracy (blindly guessing the most common class each time)
* precision = (# true positives) / (# true positives + false positives)
* recall = (# true positives) / (# true positives + false negatives)
* F-score = 2 \* precision \* recall / (precision + recall)
* Area under the Receiver-Operating-Characteristic Curve
	* also known as AUC of ROC
	* a curve measuring the growth of the true-positive-rate (y-axis) relative to the false-positive-rate (x-axis) as a model's threshold is reduced
* Python: [scikit-learn](http://scikit-learn.org/stable/modules/model_evaluation.html)

---

# Checking for Assumptions and Correcting Data
* In general, it's almost always assumed that the quantity of data exceeds the quantity of features. Most methods will be ineffective if this is not true, and and some methods may have impossible computations.

## Idependently and Independentically Distributed Observations
* fundamental assumption of almost all statistical learning methods

## Normality
* histogram to make sure there is no skew
* QQ-plot against Normal Distribution - if straight line, normal
* Shapiro-Wilk Test (if p < .05, then not a normal distribution)
	* R: `shapiro.test()`
	* Python: `scipy.stats.shapiro()`
* Fixing Skew
	* Transformations
		* Positive / Right-Tail
			* Log, Square-Root, or Reciprocal
				* requires data to be positive - if necessary, add a constant to all observations
		* Negative / Left-Tail
			* same as above, but reverse scores by subtracting each observation from the largest observation
	* Robust Methods
		* Bootstrappings

## Homogeneous Variance (Homoscedasticity)
* plotting y values should show that the variance of the residuals' distance from the regression line does not change with respect to x
* Levene's Test or F-Test/Variance Ratio (if p < .05, then variance is heterogeneous)
	* R: `levene.test()` and `var.test()`
	* Python: `scipy.stats.levene()` and [`scipy.stats.f()`](http://stackoverflow.com/questions/21494141/how-do-i-do-a-f-test-in-python)

## Feature Selection

## Nonlinear relationships

## The Problem with Hypothesis Testing
* for large enough samples, alternate hypothesis will always be true
* because of this, we shouldn't rely exclusively on hypothesis testing to determine whether our data satisfies the model's assumptions
