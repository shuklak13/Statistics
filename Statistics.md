# Hypothesis Testing
* to find a p-value, use the `pnorm()` or `pt()` functions
* to do a hypothesis test, you can either...
	1. check if the p-value is less than alpha, or...
	2. use one of the many `*.test()` functions (like `z.test`, `t.test`, and `cor.test()`)


# Correlation
## Pearson's Correlation Coefficient r
* `r = cov(x,y)/s(x)s(y)`
* [-1, 1]
* measures the linear relationship between two variables
* Assumptions:
	* both variables are normally distributed
		* large sample, or...
		* sample is normally distributed
	* if not met, use Bootstrapping or another correlation measure
* Testing significance
	* We can transform r into a t-statistic (in order to perform a Hypothesis Test) as follows:
		* `t = r * sqrt(N-2) / sqrt(1-r^2)`
		* in R, use `lm.beta()` from the R package
	* of course, you can just use `cor.test` in R
* "Bivariate Correlation" - doesn't account for the effect of additional variables

#### Dichotomous Correlations
* one of the two variables is binary
* 2 Categories:
	1. point-biserial correlation - dichotomy is discrete (pregnant or not)
		* this is the same as Pearson's Correlation Coefficient r
	2. biserial correlation - dichotomy has underlying continuous distribution (passing vs failing exam)
		* `biserial correlation = r * sqrt(pq)/ y`
			* p and q are the proportions of the two values of dichotomous variable x
			* y is the value of the normal distribution when p% of the distribution is to the left and q% is to the right
		* compute with `polyserial()` in R
		* not supported by `cor.test()`, so you have to program the hypothesis test yourself

## Spearman Correlation Coefficient
* nonparametric - useful alternative to Pearson's if your data violates parametric assumptions
* measures the strength of a monotonic relation, rather than a linear relation
* as with Pearson's, use `cor.test` to test significance

## Coefficient of Determination R^2
* proportion of y's variability that is shared by x = square of the correlation between the observed and predicted y's
* `R^2 = SS_T - SS_M / SS_T`
	* where SS_T is the sum of squared residuals from the mean and SS_R is the sum of squared residuals from the regression line
	* computed differently for single regression and multiple regression
* increases with every variable you throw in the model, which can falsely reward models with lots of useless variables

### Ways to Penalize for Having Too Many Useless Variables
* we want to minimize both SSE and # of variables

#### Adjusted Coefficient of Determination
* can be used as a stand-alone measure, trained on different datasets
* `Adjusted R^2 = 1 - (1-R^2)*(n-1)/(n-p-1)`

#### Akaike Information Criterion (AIC)
* `AIC = n*ln(SSE/n) + 2k` where k is the # of variables

#### Bayesian Information Criterion (BIC)
* `BIC = k*ln(n) + 2*ln(L)`
	* where L is the max value of the model's likelihood function
* both AIC and BIC can't be used as a stand-alone measure - can only be used to compare two models with the same data

## F-Ratio
* ratio of variability the model can't explain vs variability the model can explain
* Mean Sum of Squared Error from the Regression Model / Mean Sum of Squared Error from the Mean
* Close to 0 means the model is very good, Close to 1 means the model is no better than just guessing the mean every time

## In R?
* you can get R^2, Adjusted R^2, the F-Ratio, and the p-value of a model using the `summary()` function in R
* you can compare multiple models using `anova()`

## Partial Correlation
* accounts for the effects of additional variables, unlike Bivariate Correlation
* does a better job of measuring two variables' relationship by holding the other variables constant
* use `pcor` and `pcor.test` in R

## Computing the Difference between Correlations
* if independent
	* xy vs xy in the different sample (Ex: male vs female)
	* `z-stat = (z1-z2)/sqrt(1/(n1-3) + (1/(n2-3)))`
* if dependent
	* xy vs zy in the same sample
	* `t-stat = (r_xy-r_zy) * sqrt[(n-3)(1+r_xz) / 2(1-r_xy^2-r_xz^2-r_zy^2+2r_xy*r_xz*r_zy)]`


# Regression
* prior to creating regression models, it's a good idea to...
	* identify outliers and influential points, and perhaps exclude them **if** *you have a good reason to do so*
	* transform data to create linear relations that satisfy regression assumptions

## Outliers
* standardize your data points' residuals (Subtract mean(resid), Divide sd(resid)) to convert to z-scores
	* `rstandardize()`
* 99.9% of data should lie within z=[-3.29, 3.29], so values outside of this range are most likely extreme outliers or errors
* less than 1% of data should lie outside of z=[-2.58, 2.58], and less than 5% of data should lie outside of z=[-1.96, 1.96] - more than this, and your model is probably a poor fit for your data

## Influential Points
* points that have an unusually large influence on the model's form
* either are outliers, or have large leverage (meaning their x-values are far away from the x-values of other points)
	* find leverage with `hatvalues()`
* Ways to Identify
	* Studentized Residual = `DFFit(x) / StandardError`
		* where DFFit is the difference in the prediction of x between the model that includes x and the model that does not include x
			* `dffits()`
		* only measures how much a point influences its own prediction
		* `rstudent()`
	* Cook's Distance
		* measures how much a point influences all predictions
		* if Cook's Distance > 1, it is probably an influential point
		* `cooks.distance()`

## Methods of Regression
* Hierarchical
	* add known predictors (those that have worked well in the past) first, before adding new predictors (those you are testing)
* Forced Entry
	* add everything simultaneously
* Stepwise Methods
	* forward
		* start with nothing, and greedily add predictors to optimize performance
	* backward
		* start with everything, and greedily remove predictors to optimize performance
		* typically considered better than forward stepwise regression because ___
	* because Stepwie is greedy, they're not guaranteed to give the best performance
* All-Subsets Method
	* from all potential subsets of predictors, choose the one that optimizes performance
		* unlike Stepwise, All-Subsets IS guaranteed to give the best performance
	* the most "complete", but also takes the most amount of time

#### How to Choose a Method?
* add variables in order of importance
	* determine "importance" by the prevalence of previous literature and prior beliefs
* Stepwise and All-Subsets are vulnerable to random sample variation, so they may not always give the same regression models
	* Because of this, some people prefer Hierarchical and Forced Entry

## Regression Assumptions
* No Multicollinearity
	* the predictors do not correlate with each other
	* test this assumption by...
		* creating a correlation matrix on the predictors (ballpark method)
		* checking if the Variance Inflation Factor is high (smarter method)
			* >10 is definitely multicollinear, >5 is cause for concern
			* `vif()`
	* the variance of the predicted variable should be constant w.r.t. the predictor variables
* Errors are Normally Distributed, Independent, and have Homoscedasity
	* No Autocorrelation = the errors of one observation should not influence the errors of the next observation
	* test this assumption with the Durbin-Watson Test (test stat = 2 means no correlation, 0 means positive, 4 means negative)
		* order matters - you must keep the data in the original order for Durbin-Watson to be useful
		* `durbinWatsonTest()`
	* Plotting
		* test Homoscedasity by plotting fitted values vs residuals with `plot()`
		* test Normality of Errors by plotting residuals with `hist()` and `qqnorm()`

## Rules of Thumb - How Many Observations Do I Need?
* To Test the Overall Fit of Your Regression Model: `50 + 8 * (# variables)``
* To Test the Predictors: `104 + (# of variables)`

## Robust Regression
