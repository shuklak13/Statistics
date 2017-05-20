# Hypothesis Testing
* to find a p-value, use the `pnorm()` or `pt()` functions
* to do a hypothesis test, you can either...
	1. check if the p-value is less than alpha, or...
	2. use one of the many `*.test()` functions (like `z.test`, `t.test`, and `cor.test()`)

## T-Test
* Repeated Measures Design: using the same subjects multiple times, rather than each observation being on a new subject (Ex: before-and-after experiments)
* for these experiments, we need to use the Matched-Pairs T-Test (AKA Dependent-Means T-Test) instead of the traditional Independent-Measures T-Test (AKA Independent-Means T-Test).
* `t.test()` in R uses Welch's t-test, which corrects for heteroscedasity by reducing the degrees of freedom
	* Tall - `t.test(group ~ measure, data)`
	* Wide - `t.test(groupMeasure1, groupMeasureB, paired)`
* `yuen(groupMeasure1, groupMeasureB, trimmedPercentage)` from WRS2 is used to trim outliers before the t-test, or `yuend()` if dependent
	* `yuenbt(groupMeasure1, groupMeasureB, trimmedPercentage, nboot)` to use bootstrapping, or `ydbt()` if dependent
* `pb2gen(groupMeasure1, groupMeasureB, nboot)` from WRS2 is used for M-estimation with bootstrapping, or `bootdpci()` if dependent
* what if you have more than 2 hypotheses? ANOVA

### Independent T-Test
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

### Dependent T-Test
* pretty much same as Independent, except that instead of being concerned with difference of means, we're concerned with means of differences
