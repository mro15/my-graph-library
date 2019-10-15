# Graph of words experiments

This project contains the  implementation of my tests with graph of words.

# Datasets
- Polarity:
	- 1000 negative reviews
	- 1000 positive reviews
- IMDB
	- 25000 negative reviews
	- 25000 positive reviews

# Experiments

## First experiment
In my first experiment I created graphs of words with sliding window size is 4, 7, 12 and 20.
After build a graph for each review, the node embeddings were extracted using **node2vec**.
The node embeddings are fed into a CNN for sentence classification.
We perform stratified 10-fold cross validation and report the mean and standard deviation.
Student t-test and Wilcoxon test are performed.
