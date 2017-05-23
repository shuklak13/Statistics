# What's this going to tell me?
This master cheatsheet will tell you...

1. some different statistical problems (regression, classification, etc.)
2. the techniques used to model solutions for those problems
3. the assumptions those techniques require
4. the metrics that can be used to measure the effectiveness of the resulting models
5. the pros and cons of each technique
6. how to use those techniques in [R and/or Python](http://mathesaurus.sourceforge.net/r-numpy.html)

## Python vs R
* although this cheatsheet supports both, many statistical techniques are currently not supported by Python. In this case, you have two options:
	1. Implement the technique yourself, or...
	2. Use [rpy2](https://pypi.python.org/pypi/rpy2) to call R code from Python
* if you want to use Python, you should be aware of the following libraries:
	* pandas - data.frame
	* sklearn - machine learning
	* statsmodel - traditional statistics

---

# Regression
## Linear Regression
### Ordinary Least-Squares Linear Regression
* R: `lm(y ~ x)`
* Python: `linear_model.LinearRegression().fit(x)` from sklearn, or `statsmodel.formula.api.ols(formula, data).fit()` from statsmodel

### Robust Linear Regression

### Polynomial Least-Squares Regression
* R: `lm(y ~ poly(x, 3))` to perform a regression of the form `y = ax^3 + bx^2 + cx + d`
* Python: `linear_model.LinearRegression.fit(PolynomialFeatures(degree=2).transform(X))` from sklearn and sklearn.preprocessing

## Decision Tree Regression
* R: `rpart(y ~ x, "anova")` from rpart
* Python: `tree.DecisionTreeRegressor().fit(x, y)` from sklearn

### Random Decision Forest Regression
* R: `randomForest()` from randomForest
* Python: `RandomForestRegressor()` from sklearn.ensemble

## Neighbor Regression
* R: `knn.reg()` from FNN
* Python: `KNeighborsRegressor()` and `RadiusNeighborsRegressor()` from sklearn.neighbors

## SVM Regression
* [R: `svm(y ~ x)` from libsvm](https://cran.r-project.org/web/packages/e1071/vignettes/svmdoc.pdf)
	* Note that SVM regression and classification both use the same code! The function determines whether to perform regression or classification depending on whether y is continuous or categorical.
* [Python: `LinearSVR`, `SVR`, and `NuSVR` from sklearn](http://scikit-learn.org/stable/modules/svm.html)

## Hierarchical Linear Model
* Multiple levels [(ex: school, district, state, country)](https://stats.stackexchange.com/questions/63621/what-is-the-difference-between-a-hierarchical-linear-regression-and-an-ordinary). For any observation, gives most weight to lowest level if large enough sample. If too small, gives more weight to the level above.
* [NOT to be confused with Hierarchical Regression, which is just creative a successive chain of regression models adding more predictors each time](http://www.theanalysisfactor.com/confusing-statistical-term-4-hierarchical-regression-vs-hierarchical-model/)
* R:
* Python:

## Ride Regression AKA Weight Decay AKA Tikhonov Regularization
* [does *not* give an unbiased estimator](https://onlinecourses.science.psu.edu/stat857/node/155)
* penalizes large coefficients
	* tradeoff beteen penalty term and residual-sum-of-squares (better coefficients give less error)
* good for dealing with multicollinearity
* R:
* Python:

## Lasso

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
* Python: [scikit-learn](http://scikit-learn.org/stable/modules/model_evaluation.html) or [`model.fit().summary()` from statsmodel](http://www.statsmodels.org/devel/generated/statsmodels.regression.linear_model.RegressionResults.summary.html)

---

# Classification
## Logistic Regression
* Assumptions:
* Advantages:
	* relatively robust: no assumptions are made about the data's distribution
* R: `glm(family = binomial())` from the mlogit, or `mlogit()` if the output consists of multiple classes
* Python: `LogisticRegression().fit()` from sklearn.linear_model, or [`Logit(y, x).fit()` or `GLM(family=families.Gamma())` from statsmodels.api](http://www.statsmodels.org/stable/examples/notebooks/generated/glm.html)

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

### Random Decision Forests
* R: `randomForest()` from randomForest
* Python: []`RandomForestClassifier()` from sklearn.ensemble](http://scikit-learn.org/stable/modules/ensemble.html#forest)

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

## Naive Bayes
* R:
* Python:

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

# Clustering

## K-Means
* R:
* Python: [`KMeans(n_clusters)` from sklearn.cluster](http://scikit-learn.org/stable/modules/clustering.html#k-means)

## Hierarchical/Agglomerative Clustering
* R:
* Python: [`AgglomerativeClustering().fit(data)` from sklearn.cluster](http://scikit-learn.org/stable/auto_examples/cluster/plot_digits_linkage.html#sphx-glr-auto-examples-cluster-plot-digits-linkage-py)

---

# Preprocessing: Checking for Assumptions and Correcting Data
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
		* Bootstrappingsle
* Z-scores (for normally-distributed data)
	* 99.9% of data should lie within z=[-3.29, 3.29], 99% within z=[-2.58, 2.58], and 95% within z=[-1.96, 1.96]
	* 99.7% of data should lie within 3 standard deviations, 95% within 2 standard deviations, and 68% within 1 standard deviations
	* if not, your data does not follow a standard normal distribution
	* R: `scale()`
	* [Python: `preprocessing.scale()` from sklearn](http://scikit-learn.org/stable/modules/preprocessing.html#preprocessing-scaler)

## Multicollinarity

## Homogeneous Variance (Homoscedasticity)
* plotting y values should show that the variance of the residuals' distance from the regression line does not change with respect to x
* Levene's Test or F-Test/Variance Ratio (if p < .05, then variance is heterogeneous)
	* R: `levene.test()` and `var.test()`
	* Python: `scipy.stats.levene()` and [`scipy.stats.f()`](http://stackoverflow.com/questions/21494141/how-do-i-do-a-f-test-in-python)

## Feature Selection and Dimensionality Reduction
* Why?
	* avoid curse of dimensionality
	* reduce risk of overfitting
	* number of features should be less than number of observations

### Feature Selection
* get rid of redundant/irrelevant features
* compare models to maximize performance metrics (Adjusted R^2, AIC, BIC, etc.)
* Regression
	* Stepwise
		* backways tends to perform better than forward
		* greedy, locally optimized, fast
		* R: `step()`
		* Python: [here's a forward stepwise regression function for `statsmodel`](http://planspace.org/20150423-forward_selection_with_statsmodels/)
	* All-Subsets
		* globally optimized, slow
		* R: `leaps()` from leaps

### Dimensionality Reduction
#### Low Variance Threshold
* remove all features which have insufficient variance
* R:
* Python: [`VarianceThreshold(threshold).fit_transform(data)` from sklearn.feature_selection](http://scikit-learn.org/stable/modules/feature_selection.html#removing-features-with-low-variance)

#### Statistical Tests
* check the p-value to determine if an independent variable has an impact on the dependent variable
* R:
* Python [various functions in sklearn.feature_selection] (http://scikit-learn.org/stable/modules/feature_selection.html#univariate-feature-selection)

#### Principle Components Analysis
* reduces features space to vectors representing linear combination of features
* decompose a feature space into orthogonal vectors that maintain the maximum amount of the original space's variance
* R:
* Python: [`PCA(n_components).fit(data).transform(data)` from sklearn.decomposition](http://scikit-learn.org/stable/modules/decomposition.html#pca)

## Nonlinear relationships

## The Problem with Hypothesis Testing
* for large enough samples, alternate hypothesis will always be true
* because of this, we shouldn't rely exclusively on hypothesis testing to determine whether our data satisfies the model's assumptions

## Outliers and Influential Points
* These should be removed **if** you can conclude that they are data entry errors, or they come from a different population than the rest of the sample
* Investigate potential outliers/influential points using multiple metrics - don't make hasty conclusions
* Note that transformations can modify outliers/influential points! For maximum coverage, look at outliers/influential points before and after you transform.
* [Python: `influence_plot(prestige_model, criterion)` from `statsmodels.graphics`](http://www.statsmodels.org/0.8.0/examples/notebooks/generated/regression_plots.html)
	* plots the influence of each observation, using the specified criterion (Cook's Distance or DFFITS)
* You can use visual indicators like scatterplots or residual plots to determine outliers and influential points. You can also use statistical metrics, given below.
* [R: `influence.measures(model)`](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/influence.measures.html)
	* gives you standardized, studentized residuals, dfbetas, Cook's Distances, and hat values for each observation used to train the model
* [Python: summary_table() from statsmodels.stats.outliers_influence ](http://www.statsmodels.org/0.6.1/_modules/statsmodels/stats/outliers_influence.html)
	* contains functions for VIF, studentized residuals, dffits, dfbetas, Cook's Distance,

#### Outliers
* points that fit the model poorly
* [Studentized Residual](https://stats.stackexchange.com/questions/22653/raw-residuals-versus-standardised-residuals-versus-studentised-residuals-what)
	* residual, divided by sample standard error
	* R: `rstudent()`

#### Influential Points
* exert an unusually large influence ont he model's coefficients
* [Leverage AKA Hat Value](https://en.wikipedia.org/wiki/Leverage_%28statistics%29)
	* measures how far an observation's independent variables deviate from their mean
	* if an observation is more than 2x or 3x of the average leverage `(k+1)/N`, then it is probably an influential point
	* R: `hatvalues()`
* [DFFITS](https://en.wikipedia.org/wiki/DFFITS)
	* the difference in the prediction of an observation x, between the model that includes x, and the model that does not include x
	* equal to the studentized residual, scaled by leverage
	* R: `dffits()`
* Cook's Distance
	* measures how a particular observation influences the predictions of *all* observations in the training set (unlike DFFITS, which is only measures the influence on the same observation)
	* conceptually equivalent to DFFITS
	* [if greater than 1 or 4/N, probably an influential point](https://en.wikipedia.org/wiki/Cook%27s_distance#Detecting_highly_influential_observations)
	* R: `cooks.distance()`
* DFBeta
	* the influence that one particular observation had on one particular predictor
	* R: `dfbeta(model)` returns the DFBeta matrix
		* `dfbeta(model)[i]` returns the DFBetas for the i'th observation
		* `dfbeta(model)[,j]` returns the DFBetas for the j'th variable
	* [values greater than 1 or 2/sqrt(n) are probably influential points](http://www.albany.edu/faculty/kretheme/PAD705/SupportMat/DFBETA.pdf)
