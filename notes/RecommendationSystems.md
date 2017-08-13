Source: ["Evaluating Recommendation Systems"](https://www.microsoft.com/en-us/research/publication/evaluating-recommender-systems/) by Guy Shani and Sela Gunawardana at Microsoft Research

# Metrics
* accurate results
	* confidence (a recommender with accurate predictions and high confidence is better than one with accurate predictions and low confidence)
* diverse results
* serendipitous results (predictions that are not only accurate, but also surprising - as opposed to "obvious" predictions)
* fast response
* adaptivity to new information (new genres, shifting user preferences)
* respect for privacy
	* recommendations should not be so specific as to allow a third party to guess the browsing history of users
	* Ex: if user A is one of only a handful of people interested in topic X, and if user B browses topic X and gets a recommendation for topic Y, then user B can guess that user A is interested in topic Y
* profitability
* increased user engagement
* usefulness for wide range of users/items (obscure interests, cold-start)

# Experiment Design
1. Determine a hypothesis
2. Control all variables except those intended to be tested
3. Consider the scope at which the experiment's results can be generalized
	* the more diverse the dataset, the more an experiment's results can be generalized
