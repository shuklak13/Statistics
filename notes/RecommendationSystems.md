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
	* click rate
		* Pro: more adverising
		* Con: promotes clickbait
	* session duration
		* usually a better metric than click rate
* usefulness for wide range of users/items (obscure interests, cold-start)

# Experiment Design
1. Determine a hypothesis
2. Control all variables except those intended to be tested
3. Consider the scope at which the experiment's results can be generalized
	* the more diverse the dataset, the more an experiment's results can be generalized

# Interesting Insights about the Youtube Recommendation System
* [2010 paper](https://www.researchgate.net/publication/221140967_The_YouTube_video_recommendation_system)
	* to compute how related two videos are, the following formula is used
		* `r_ij = c_ij / (c_i * c_j)`
			* where `c_i` and `c_j` are how many times videos i and j were watched, while `c_ij` are how many times videos i and j were both watched in the same session
	* 3 different components are used for ranking - video quality, user specificity and diversification
		* Video Qualtiy: videos of high # of views or thumbs-ups are ranked higher
		* User Specificity: videos relevant to the user are ranked higher
		* Diversification: in any batch of candidates, videos with low similarity are selected
	* To optimize speed, recommendations are not computed dynamically. Instead, batches of recommendations are computed several times per day. Only a subset of those recommendations are shown at a time. When the user visits their homepage, a different subset is shown, to give the illusion of new recommendations being generated.
	* Data is stored in Google BigTable
	* metrics
		* click rate
		* long click rate (clicks that led to long watches)
		* session length
		* time to first long watch
* [2016 paper](https://research.google.com/pubs/pub45530.html)
	* crafting a recommendation system is "more art than science"
	* Youtube has switched to using neural nets powered by Tensorflow
		* video features (inspired by continuous bag of words) and user features (geographic region, gender, age, etc.) are input into these neural nets
		* one of the biggest advantages of neural nets is that it is easy to add new, arbitrary features (both continuous and categorical)
	* Pipeline
		* **Millions of Videos**
		* **Candidate Generation Neural Net** goes through those millions of videos and selects hundreds of them based on user prefernces
		* **Ranking Neural Net** selects dozens from those hundreds of videos via feature engineering
		* Those recommendations are then **presented** to the user's homepage in a manner similar as before
	* Freshness
		* Problem: users significantly prefer newer videos; however, machine learning algorithms are biased towards older content
		* Solution: add "age" as a feature to the model
