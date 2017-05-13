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


## Coefficient of Determination R^2
* proportion of y's variability is shared by x
* increases the more variables you throw in the model, which can falsely reward models with lots of variables

#### Adjusted Coefficient of Determination
* penalizes R^2 for points that don't fit the model

## Spearman Correlation Coefficient
* nonparametric - useful alternative to Pearson's if your data violates parametric assumptions
* measures the strength of a monotonic relation, rather than a linear relation
* as with Pearson's, use `cor.test` to test significance

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
