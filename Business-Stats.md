# Standard Techniques

### Segmenting data into groups
Can be done **manually** or via **clustering**

### Visualization
* X-vs-Y graphs	- nice for 2 vars
* Radar Charts - nice for multiple vars

### Standardize variables
subtract mean, divide standard deviation

### Regression
* Can be used for *prediction* and *correlation*

##### Statistical Significance	(p-values)
* useful for...
	* finding correlations	(but NOT causations!)
	* removing insignificant predictors

##### Residual Analysis Plots
* QQPlot against Normal Distribution
	* make sure errors follow a normal distribution (45degree line)
* Y-value vs Residual on a dotplot
	* to make sure there is no relation between the two

##### Measuring Performance
	* R^2
	* RMSE	(requires testing data)

##### Survival Analysis	(subcategory of regression)
* how long until something happens
* "right-censored": we know past but not future
* Output = Survival Function and Hazard Function
	* Survival = CDF
		* probability of event not occurring up to some time for some individual
	* Hazard = PDF
		* probability event will occur at some time, if individual has survived until that time
* Differences from traditional regression
	* Survival never gives negative y-values
	* Survival has to deal with Right-Censoring (some data won't last the entire timespan)
* Business Application Examples
	* customer/employee retention
	* equipment maintenance

##### Time Series
* Characteristics to Consider
	* Aggregate Trends	(up or down)
	* Seasonality/Cycles	(repeating highs/lows)
	* Variance			(constant or not)
* AR(1) - Autoregressive of Order 1
	* use value of previous time to find value of present time
	* errors assume normally distributed w/ constant variance
* Differences from traditional regression
	* data has an order
	* autocorrelation
		* observations close in time will be more similar than those farther apart

### Classification
* Logistic Regression
	* good if two categories (linear boundary)
* Decision Tree
	* good if multiple categories (nonlinear boundary)
* Measuring Performance
	* Precision/Recall	(requires test set)

### Tools
as a data scientist, don't be afraid to switch language or software according to your needs
* R
* Python
* Excel

# Business Use Case Examples

### Managing Stocks of Products

##### Prediction / Regression
* Y = SKU = Stock-Keeping-Unit
* X = Volatility of Sales (StandDev(Sales)/Mean(Sales))

##### Visualization
* X-vs-Y Graph (Sales, Variability)
* Groups
	* Horses - High Sales, Low Variability
		* b/c we can predict these well, we should forecast their demand and use that to keep them in stock
	* Bulls - High Sales, High Variability
		* hard to make a universal rule for all products - evaluate on a case-by-case basis
	* Crickets - Low Sales, High Variability
		* usually don't sell well, but may sell well sometime
		* made-to-order (don't stock ahead of time, only as needed)


### Employee or Customer Retention

##### Regression
* Y = # of employees who left
* X = value of a particular feature (Ex: # of years at company)
* Purpose
	* predict who will leave company in the future, so we can take preventative action
	* We want to target employees with the highest Expected Loss
		* Expected Loss = (Probability of Leaving)(Value to Company)
* could do Survival Analysis

##### Clustering
* Y = Satisfaction
* X = Features of Employee (including performance)
* Groups
	* Low-performers (low satisfaction, low performance)
		* not worth retaining
	* Burned-out (low satisfaction, high performance)
		* we need to act to retain these people!
	* High-potential (high satisfaction, high performance)
		* we don't know why they left, so it's difficult to act here

### Customer Segmentation (Ex: Telecommunications)
##### Clustering
* Groups
	* Old	(higher age, less usage)
	* Teens	(few calls, many texts)
	* Young Adults	(few calls, many texts)
	* 30s
	* 40s
	* Callers	(high calls, few texts)
* Merge Groups
	* some clusters might differ only in attributes * irrelevant to business problem
	Ex:	Teens + Young Adults, 30s + 40s

### Credit Scoring
##### Prediction / Regression
* Y = Credit Score
* X = Features of Individual (including age, education, gender, ethnicity, debt)
* Purpose
	* find customers who do not yet have good credit score, but should be acquired as clients
