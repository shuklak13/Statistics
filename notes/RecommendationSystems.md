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

# Youtube Recommendation System
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

# [Twitter's "WTF" (Who-to-Follow) Recommendation System](https://stanford.edu/~rezab/papers/wtf_overview.pdf)

* WTF was created in 2010
* 2 Goals:
	* "Similar to" (Bob and Tom both like sports, they'll follow each other)
	* "Interested in" (Bob likes sports; Bob will be interested in ESPN)
* Twitter's user base can be visualized as a directed graph
	* as of 2012, the graph has over 20 billion edges
* interestingly, Twitter's recommendation system model is based on the assumption that the entire graph of users will fit in memory on a single machine
* Twitter's Technology Stack (as of 2010):
	* Cassovary, Twitter's open-source in-memory graph processing engine
	* FlockDB, Twitter's open-source graph database built on MySQL (no longer supported)
	* Gizzard, Twitter's open-source sharding framework to create distributed, eventually-consistent datastores (no longer supported)
* OLTP vs OLAP
	* OLTP = online transaction processing
		* short workloads dominated by seek times
		* Cassovary
	* OLAP = online analytical processing
		* sequential scans over large amounts of data
		* FlockDB + Gizzard
	* Because these two types of data processing are so different, modern architectures often separate their workloads to give optimal performance for each type of operation
* Why not Hadoop?
	* MapReduce jobs have high startup costs, giving a large lower-bound on operation time
	* Can be a poor choice for extremely unevenly distributed data, since some reducers will have significantly more work than others
* Why not distributed?
	* Short deadline required faster solution
	* It's totally possible to store the entire Twitter database on one machine
	* With only one machine, there is no network latency, greatly improving speed
* Relies on higher-end hardware (unlike most distributed systems)
	* dual quad core processors, 144GB RAM
	* with 2 billion users, each user's id stored as a 32-bit (4-byte) int, each edge require 8 bytes (2 id's)
		* a 72GB computer should be able to hold 8 billion edges
		* as Twitter's user base increases, so does the size of RAM of top-end computers
* Production Flow
	* Twitter graph from FlockDB is imported daily
	* Recommendations are being generated constantly on Cassovary servers
		* Users are put in a queue - those who have consumed most of their recommendations are given higher priority for recommendation generation
		* New users are also given hight  priority to overcome the cold-start problem
* SALSA = Stochastic Approach for Link-Structured Analysis
	* an algo initially used for web search ranking (same family of random walk algos as PageRank)
	* the basis of Twitter's user's recommendations
	* split graph into two sides (bipartite) - "hubs" (followers) and "authorities" (follows)
	* each step in the random walk goes over two links - one from the hubs to the other authorities, and one back from authorities to hubs
	* after the graph is constructed, several iterations of random walk are executed in a manner similar to PageRank to assign scores to both sides
	* after scores are assigned, the scores of the authorities as used for "interested-in" recommendations, and the scores of the hubs are used for "similar-to" recommendations
* In production, Twitter uses an ensemble of recommendation algorithms (~20 algos)
* Evaluation of the Algorithm
	* Things to Test
		* different recommendation algorithms
		* different algorithm parameter values
	* Types of Experiments
		* Online A/B testing on users
			* slow - good for evaluating models that did well on test environments
		* Offline retrospective experiments
			* much faster to conduct, typically done to decide which models should be deployed for A/B testing
	* Metrics
		* Follow-Through-Rate (FTR)
			* (# follows)/(# recommendations given)
		* Engagement-Per-Impression (EPI)
			* how much users are engaged with recommended follows
			* gives a measure of the quality of recommendations, but takes longer to complete test
		* 