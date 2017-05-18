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
	* identify outliers and influential points, and perhaps exclude them **if** *they can be determined to be data entry errors or from a different population than the rest of the sample*
	* transform data to create linear relations that satisfy regression assumptions
* remember - if you have multiple models that seem equally valid but give different conclusions, then your data is insufficient to answer your question unambiguously
* the significance of individual predictors can be measured via T-test
* `lm(equation, data)` from the stats package
	* you can model interaction terms in your equation using `term1:term2`

## Outliers and Influential Points
* Outliers
	* standardize your data points' residuals (Subtract mean(resid), Divide sd(resid)) to convert to z-scores
		* `rstandardize()`
	* 99.9% of data should lie within z=[-3.29, 3.29], so values outside of this range are most likely extreme outliers or errors
	* less than 1% of data should lie outside of z=[-2.58, 2.58], and less than 5% of data should lie outside of z=[-1.96, 1.96] - more than this, and your model is probably a poor fit for your data
* [Influential Points](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/influence.measures.html)
	* points that have an unusually large influence on the model's form
	* either are outliers, or have large leverage (meaning their x-values are far away from the mean x-values)
	* Leverage
		* `hatvalues()`
		* ranges from 0 to 1
		* on average `(k+1)/N`
			* more variables => larger average leverage
			* more observations => smaller average leverage
			* as is evident, if the # of variables and observations is almost equal, then each point will have a huge amount of influence! This is bad
		* if an observation is 2-3x more than the average leverage, it is probably an influential point
	* Ways to Identify
		* Studentized Residual = `DFFit(x) / StandardError`
			* where DFFit is the difference in the prediction of x between the model that includes x and the model that does not include x
				* `dffits()`
			* only measures how much a point influences its own prediction
			* `rstudent()`
			* similar to Standardized Residuals, only 5% should lie outside 1.96, only 1% should lie outside 2.58
		* Cook's Distance
			* measures how much a point influences all predictions
			* if Cook's Distance > 1, it is probably an influential point
			* `cooks.distance()`
		* DFBeta
			* the difference that one particular observation had on one particular predictor
			* so, there is `(# observations)x(# predictors)` DFBeta values
			* `dfbeta(model)` returns the DFBeta matrix
				* `dfbeta(model)[i]` returns the DFBetas for the i'th observation
				* `dfbeta(model)[,j]` returns the DFBetas for the j'th variable
			* DFBetas should be less than 1
* Summary
	* Outliers
		* points that the model fits poorly
		* identified via studentized/standardized residuals and deviance statistics (Logistic Regression)
	* Influential Points
	 	*  exert an unusually large influence on the model
		* identified vai Cook's Distance, DFBeta, and leverage statistics

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
		* typically considered better than forward stepwise regression because forward stepwise regression may exclude predictors with suppressor effects (significant, but only when another variable is held constant)
	* because Stepwise is greedy, they're not guaranteed to give the best performance, but it can give you a reasonably fine-tuned model in a reasonable amount of time
	* you can also step both ways (every time you remove an element, see if you can add one) which generally performs better
	* `step()` from the stats package
* All-Subsets Method
	* from all potential subsets of predictors, choose the one that optimizes performance
		* unlike Stepwise, All-Subsets IS guaranteed to give the best performance
	* the most "complete", but also takes the most amount of time
	* `leaps()` from the leaps package

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
			* `vif(model)`
	* the variance of the predicted variable should be constant w.r.t. the predictor variables
	* if multicollinearity exists, you should omit one of the variables
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
* an alternative to Ordinary Least Squares Regressions
* Robust: a model that is useful even if its assumptions are not met
* M-Estimation
	* a class of estimators which are the optimization of an estimating function (the maximum in a MLE, the zero in a derivative
 	* good if you have outliers/high-leverage points in your data that you can't remove
	* stands for "maximum likelihood-type"
	* "best" (lowest-variance) unbiased estimator
	* M-Estimation "weights" data based on how nicely-behaved it is, so that points with higher residuals have a smaller impact on the regression line
	* create some function `p(e)` of the residual `e` where `p` is...
	 	* always nonnegative
		* symmetric about `e=0`, and equals 0 when `e=0`
		* monotonically increases away from `e=0` (as e gets farther from 0, p increases)
	* Ex: least-squares and maximum-likelihood are special cases of this
	* `rlm` from the MASS package
* [Sandwich, Meat, and Bread](https://cran.r-project.org/web/packages/sandwich/index.html)
	* the variance of your residuals is not constant (Heteroscedasity)
* Bootstrapping
	* the variance of your residuals is not constant (Heteroscedasity)
	* you have outliers in your data that you can't remove

## Contrast
* a set of variables whose coefficients add up to zero
* usually a set of dummy variables that correspond to a single categorical variable
* use `contrasts(data$variable)` to set the contrasts for a categorical variable
* use `relevel(data$variable, "baseline")` lets you reassign the baseline value of a categorical variable
* if you do regression on categorical data, R will automatically set the contrast for you

## Least-Squares vs Maximum-Likelihood forms of creating regressions
* least-squares: find the coefficients b that minimize the function of the sum of the squares of the distances between Y and Y_bar
	* by setting the derivatives of the function with respect to each coefficient to 0 and then solving for the coefficients
* maximum-likelihood: find the coefficients that maximize the likelihood
	* where the likelihood is the product of every Y's pdf, where Y is a normal distribution centered around b0+b1*X with variance equal to the variance of the error (Y-b0-b1X)

# Logistic Regression
* categorical y breaks the assumption that there is a linear relation between x and y, so we can't use the Linear Regression equation `y = a + bx`
	* use MLE to find coefficients
* `glm(family = binomial())` from the mlogit package
	* if you forget to specify the family is binomial, `glm()` will default to Gaussian (linear regression)
* can actually have more than 2 output categories (multinomial logistic regression) though traditionally we think of binomial logistic regression
	* `multinom()` from the nnet package

## Assumptions and Things to Check
* linearity between the predictors `x` and the logit `1 / (1+e^-(a+bx))`
* errors are independent
* predictors are not highly correlated
	* as with Lienar Regression, check this using VIF
* every combination of values for categorical predictors should appear at least once, and ideally more than five times
* the output does not have complete separation (there should be some points with the same x-values but different y-values)

## Evaluating the Model
* `Log-Likelihood = Sum((Y_i)ln(P(Y_i)) + (1-Y_i)ln(1-P(Y_i)))`
	* analogous to residual sum of squared errors in Linear Regression - a measure of how much of the data's variability the model does not explain
	* usually negative
	* greater Log-Likelihood => better model
* `deviance = -2LL = -2 * Log-Likelihood`
	* in R's `summary()`, this is called "Residual Deviance"
	* smaller deviance => better model
* `Likelihood Ratio = -2LL(model) = (-2LL(baseline)) - (-2LL(new))`
	* where "baseline" is just guessing the most common category every time
	* follows a Chi^2 distribution
		* compute statistical significance via `p_val = 1 - pchisq(model$null.deviance - model$deviance, model$df.null - model$df.residual)`)
			* 1st argument is the Chi^2 test statistic
			* 2nd arg is `df = # variables = (df of model using no variables) - (df of model using all variables)`
* `Pseudo R^2 = -2LL(model) / -2LL(baseline) = Log-Likelihood / -2LL(baseline) = (-2LL(baseline)) - (-2LL(new)) / -2LL(baseline)`
	* This is the Hosmer and Lemeshow Pseudo R^2 metric - there are many others, such as Cox and Snell's.
	* Logistic Regression, unlike Linear Regression, does not have a "true" R^2. The Pseudo R^2's are just convenient metrics with approximately the same meaning.
* `AIC = -2LL + 2k`
* `BIC = -2LL + 2k*log(n)`
* `fitted(model)` returns the predicted probabilities for the observations used to train the model

## Evaluating Individual Predictors
* Z-test (unlike Linear Regression, which uses the T-test)
	* `summary()`
* `Odds Ratio = (Odds after Unit Change in Predictor) / (Odds before Unit Change in Predictor) = e^b`
	* where `Odds = P(Yes)/P(No)` and `b` is the coefficient
	* under Logistic Regression, `P(Yes) = 1 / (1+e^-(a+bx))`

## Comparing Two Models
* the difference in deviances follows a Chi^2 distribution
	* compute statistical significance via `p_val <- 1 - pchisq(A$deviance - B$deviance, A$df.residual - B$df.residual)`)

## Multinomial Logistic Regression
* multiple output categories, instead of only two
	* choose one of them to be the "baseline" via `relevel()`
	* for each non-baseline category, compare its probability to that of the baseline with Binomial Logistic Regression
* `mlogit` from the mlogit package
	* to use this, you must reformat your data using `newDF <- mlogit.data(oldDF, choice = "output variable", shape
= "wide")`
* like with Binomial Logistic Regression, use p-values from `summary()` to determine the significance of predictors. Use the odds ratio via `exp(model$coefficients))` to determine the unit change in odds

# T-Test
* Repeated Measures Design: using the same subjects multiple times, rather than each observation being on a new subject (Ex: before-and-after experiments)
* for these experiments, we need to use the Matched-Pairs T-Test (AKA Dependent-Means T-Test) instead of the traditional Independent-Measures T-Test (AKA Independent-Means T-Test).
* `t.test()` in R uses Welch's t-test, which corrects for heteroscedasity by reducing the degrees of freedom
	* Tall - `t.test(group ~ measure, data)`
	* Wide - `t.test(groupMeasure1, groupMeasureB, paired)`
* `yuen(groupMeasure1, groupMeasureB, trimmedPercentage)` from WRS2 is used to trim outliers before the t-test, or `yuend()` if dependent
	* `yuenbt(groupMeasure1, groupMeasureB, trimmedPercentage, nboot)` to use bootstrapping, or `ydbt()` if dependent
* `pb2gen(groupMeasure1, groupMeasureB, nboot)` from WRS2 is used for M-estimation with bootstrapping, or `bootdpci()` if dependent
* what if you have more than 2 hypotheses? ANOVA

## Independent T-Test
* `t = [(observed difference between sample means) - (expected difference between population means under null hypothesis)]/(estimate of standard error of difference between sample means)`
	* `expected difference between population means under null hypothesis` is frequently 0, so...
		* `t = (observed difference between sample means) /(estimate of standard error of difference between sample means)`
	* The variance of a sum or difference of multiple variables is the sum of their variances, so...
		* `variance(samplingDistribution) = sd(sampling distribution)^2 = (standard error)^2 = (sd(sample)/sqrt(N))^2 = sd(sample)^2 / N`
		* `estimate of standard error of difference between sample means = sqrt[variance] = sqrt[variance(samplingDistribution1) + variance(samplingDistribution2)] = sqrt[sd(sample1)^2 / N1 + sd(sample2)^2 / N2]`
	* but, the above formula assumes the two samples are of equal size. When this is not the case, we take the pooled variance, a weighted mean of the two variances based on their sample sizes
		* `pooled variance = sp^2 = [(n1-1)sd(sampDist1)^2 + (n2-1)sd(sampDist2)]/(n1+n2-2)`
		* `estimate of standard error of difference between sample means = sqrt[sp^2 * (1/n1 + 1/n2)]`
		* use when you don't know the population variance, but you assume it is equal for both populations. So you "combine" the two samples' variances

## Dependent T-Test
* pretty much same as Independent, except that instead of being concerned with difference of means, we're concerned with means of differences


# [Linear Discriminant Analysis (LDA)](http://sebastianraschka.com/Articles/2014_python_lda.html)
* dimensionality-reduction-via projection technique, similar to PCA, commonly used in pre-processing
* differences from PCA
	* reduce the number of features of the data by finding axes which maximize separation between classes (unlike PCA, which is concerned with finding axes which maximize variance of the data)
	* supervised (class-based), unlike PCA (concerned with data only, not external classes)
* you could essentially use this as an alternative or supplement to feature selection techniques such as stepwise regression
* can also be used as a classification technique
	* assumes data is normally distributed, and that each class has identical covariance matrices
* `lda()` from MASS (has the same syntax as `lm()`)
	* returns...
		* the percentage of each group is in the data
		* the mean of each variable in each group
		*  the coefficients of the old variables used to create the linear discriminant
