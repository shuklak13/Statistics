This document will give you a basic workflow to operate with when tackling an analytical problem. "The Elements of Data Analytic Style" goes into a lot of depth on this topic.

# Workflow
1) Define a question and an experiment before you look at data.
	* failure to do this can result in [data dredging](https://en.wikipedia.org/wiki/Data_dredging) and the [multiple comparisons problem](https://en.wikipedia.org/wiki/Multiple_comparisons_problem), which can call into question the validity of your analysis
2) Create your datasets
	* You should have two datasets: a raw dataset, and a clean dataset
	* You should have documentation describing the meaning and the units of every variable
	* You should have documentation describing how the raw data was obtained
	* You should have documentation describing the transformation applied to generate the clean dataset from the raw dataset
	* Your datasets should be stored in a shared repository, and once a dataset has been published, it should not be modified.
	* Categorical and ordinal data should never be recorded as numbers - doing so can confuse future analysts into thinking your data is continuous
	* You should be aware and document any outliers before any analytics begins
	* If your dataset is so large that its size will slow down your analytics, you can use random sampling to create a smaller version of your clean dataset.
	* If you have missing values in your data, make an intelligent decision on how to handle it
		* discard those observations (can be dangerous if it leads to certain types of observations being systematically excluded from the dataset)
		* take an average of the values of the other observations (this may be good or bad depending on the situation, but note that it will reduce the variable's variability)
		* take the most common of the values of the other observations (only a good idea if one particulary values dominates the other values)
		* try to predict the missing value using a regression model created from the other variables (can be a good idea, but only if you have a large enough dataset)

## Types of Analysis
	* Descriptive - summarize and present a body of data without drawing any conclusions from it (Ex: the US census)
	* Exploratory - discover trends in existing data (note that these trends are only proposed, not confirmed; further research would need to be done to confirm the significance of these discoveries)
	* Inferential - confirm or deny some proposition (Ex: most scientific research articless)
	* Predictive - create a model that can predict the value of some variables given other variables (Ex: recommendation engines)
	* Causal - seeks to determine what happens to one variable when another variable is modified (Ex: clinical experiments)

### Exploratory Analytics
* make as many plots as possible, on as much of the data as possible, to provide as broad a view of the dataset as possible.
	* [R: ggplot2](https://www.rstudio.com/wp-content/uploads/2015/03/ggplot2-cheatsheet.pdf)
	* [Python graphing libraries](http://pbpython.com/visualization-tools-1.html)
	* if the scales of variables are very dissimilar, normalize them or perform transformations (ex: log) in order to bring them closer together
* focus on speed of analysis - not on style of cleanliness
	* those things can always be perfected later during publishing - while exploring the data, you want to be able to draw as many insights as quickly as possible
* before drawing any conclusions always consider and check for confounding variables

### Inferential
