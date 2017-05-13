# Parametric Model Assumptions

1. Normality
	* you might have to check the normality of error terms or independent variables
2. Homogeneous Variance
	* make sure that, for each independent variable x, the y-value's variance is constant across all values of x
3. Independence
	* one y-val doesn't influence another y-val
	* an example of where this assumption would be violated is a Time Series like temperature, sales, or stock market performance
4. Data is Clean and Standardized
	* not really a statistical assumption, but make sure that you are aware of any missing/incomplete/incorrect data before we dive into the analytics
	* center continuous data about its mean and divide by its standard deviation


# Normality
## Quick Rules
* Check if the sample is normally distributed.
	* If it is, then the sampling distribution probably is too.
* If you have a large sample (n>30), then according to Central Limit Theorem your sample distribution is normal, regardless of the underlying population distribution

## Skew and Kurtosis
* should both be 0
* Skew: Positive = leftward, Negative = rightward
* Kurtosis: measures extremity of outliers:
	* NOTE: What I call "Kurtosis" is actually Excess Kurtosis = (Kurtosis - 3)
	* Layman's Definition
		* Positive = pointy + heavy-tailed + more extreme outliers
		* Negative = flat + light-tailed + less extreme outliers
	* You should compare the z-score `(val/SE(val))` of the Skew and Kurtosis to the critical value to see if the skew/kurtosis is close to 0
		* Small sample: use p<.05 => 1.96
		* Large sample: use p<.01 => 2.58
		* Very large samples: don't use critical values, look at the distribution visually instead
			* Why? Standard Error decreases dramatically for large samples, rendering this formula useless for n>=200

## Ways to Empirically Test Normality
* Shapiro-Wilk Test (if p < .05, then not a normal distribution)
	*  has greatest power of all tests at a given significance (meaning it's really good at telling when the null hypothesis is false; i.e. it's really good at identifying non-normal distributions)
	* not great for large samples; use plots (Histograms, QQ) and Skew+Kurtosis instead
* QQ-plot against Normal Distribution
	* if straight line, normal
	* if S-shaped, skewed
	* if above or below straight line, kurtosis

# Homogeneity of Variance
* if you graphed the data points on a graph, the difference of y-values for any given x-value should not be significantly different

## Levene's Test
* null hypothesis: variances of different groups are equal
* like w/ Shapiro-Wilk, can reject the null hypothesis if given a large enough sample (bad for really large samples)

## F-Test AKA Variance Ratio
* `(largest variance across all groups) / (smallest variance across all groups)`
* typically more powerful than Leven's Test, but only good for normally-distributed data

# Transforming Data

## Skew
* Getting rid of Positive Skew? (convert right-tail distributions to normal)
		Log, Square-Root, and Reciprocal Transformations
		however, they require all data to be nonnegative (Square-Root) or positive (Log, Reciprocal)
			you can get this by adding a constant to all data
* Getting rid of Negative Skew? (convert left-tail distributions to normal)
		Same as above, but before transforming the data, "reverse the scores" by subtracting each value from...
			max value (least score is 0), or
			max value + 1 (least score is 1)
* Should we transform data?
		That's a difficult question. We might want to simply use a "Robust Test" instead
			Robust Test = test that does well even when its assumptions are broken
		In the end the best way is to experiment and see what gives the optimal results for your use case

## Bootstrap
	use the sample as the sampling distribution, and take many subsamples from the sample
	can be helpful if you have insufficient data to derive meaningful statistics
