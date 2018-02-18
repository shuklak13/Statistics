# Types of Analysis
* Descriptive
	* summarize and present a body of data without drawing any conclusions from it (Ex: the US census)
	* this can be useful if you are, for example, publishing a dataset for consumption by the larger analytics community
* Exploratory
	* discover trends in existing data
	* note that these trends are only proposed, not confirmed; further research would need to be done to confirm the significance of these discoveries
	* this is sometimes a step in performing one of the analyses below
* Inferential
	* confirm or deny some proposition (Ex: most scientific research articless)
* Predictive
	* create a model that can predict the value of some variables given other variables (Ex: recommendation engines)
* Causal
	* seeks to determine what happens to one variable when another variable is modified (Ex: clinical experiments)

# Workflow
1) Define a question and an experiment before you look at data.
	* failure to do this can result in [data dredging](https://en.wikipedia.org/wiki/Data_dredging) and the [multiple comparisons problem](https://en.wikipedia.org/wiki/Multiple_comparisons_problem), which can call into question the validity of your analysis
	* if you are creating a predictive model, determine the training set, testing set, and accuracy measures ahead of time
		* ideally, you should test using cross-validation
	* if you are making a causal experiment, define your treatments, control groups, and outcome variable
2) Create your datasets
	* You should have two datasets: a raw dataset, and a clean dataset
	* You should have documentation describing the meaning and the units of every variable
	* You should have documentation describing how the raw data was obtained
	* You should have documentation describing the transformation applied to generate the clean dataset from the raw dataset
	* Your datasets should be stored in a shared repository, and once a dataset has been published, it should not be modified.
	* Categorical and ordinal data should never be recorded as numbers - doing so can confuse future analysts into thinking your data is continuous
	* You should be aware and document any outliers before any analytics begins
	* If your dataset is so large that its size will slow down your analytics, you can use random sampling to create a smaller version of your clean dataset.
	* Pre-emptively determine any ways in which your sampling method might lead to biases or misrepresent the true population
	* If you have missing values in your data, make an intelligent decision on how to handle it
		* discard those observations (can be dangerous if it leads to certain types of observations being systematically excluded from the dataset - make sure to check if missingness is correlated with any variable)
		* take an average of the values of the other observations (this may be good or bad depending on the situation, but note that it will reduce the variable's variability)
		* take the most common of the values of the other observations (only a good idea if one particulary values dominates the other values)
		* try to predict the missing value using a regression model created from the other variables (can be a good idea, but only if you have a large enough dataset)
3) Exploratory Analysis
	* check your assumptions: autocorrelation, homoscedasticity, normality, etc.
		* if any of your models' assumptions are violated, take note of that and document how it will affect your results. Consider adopting an alternate model that is more robust.
	* make as many plots as possible, on as much of the data as possible, to provide as broad a view of the dataset as possible.
		* [R: ggplot2](https://www.rstudio.com/wp-content/uploads/2015/03/ggplot2-cheatsheet.pdf)
		* [Python graphing libraries](http://pbpython.com/visualization-tools-1.html)
		* if the scales of variables are very dissimilar, normalize them or perform transformations (ex: log) in order to bring them closer together
	* focus on speed of analysis - not on style of cleanliness
		* those things can always be perfected later during publishing - while exploring the data, you want to be able to draw as many insights as quickly as possible
	* before drawing any conclusions always consider and check for confounding variables
4) Perform your Analysis
	* Create your model using your data, and test how well it performed
	* Record your results, including what occurred differently than you expected and what could be changed or improved
5) Refine your problem, model, data, and/or features
	* more data usually beats better algorithms
	* more features usually beats better algorithms
	* aggregations of multiple algorithms beat individual algorithms
	* if creating a machine learning model, be aware of tradeoffs: speed/scalability vs accuracy
6) Present or Publish your Findings
	* your goal is to create a narrative. In some ways, your goal is to entertain first, educate second. If your story lacks the ability to grip your audience's emotions and make them care, then all the facts and logic in the world won't save you.
	* it's always better to go too basic in your explanations, than too advanced
	* report confidence intervals alongside any p-values; never report a p-value without the appropriate context
	* Talks
		* make the slides visually appealing, with big font and lots of self-explanatory visuals
		* use equations to highlight the technical detail of your most important model/algorithm, but do so sparingly - preferrably not more than once or twice
		* it's better to go under-time and have a memorable point than go over-Reports and overstay your welcome
	* Reports
		* R Markdown and iPython Notebooks are your best options for publishing reports for R and Python, respectively
		* keep separate directories for data, figures, R code, and text documentation
		* explain what question you're answering, what data you're using, what features are in the data, what your performance metric is, what model you're using, and what your results are
		* Reproducibility
			* record the version of any languages and packages you're using for reproducibility
			* set and record a seed before executing any nondeterministic operation
	* Packages
		* write a vignette
		* write unit tests for all your functions
