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
In my first experiment I created graphs of words with co-occurrence = 3, so the window size is 4.
After build a graph for each review, the node embeddings were extracted using **node2vec**. In this experiment for each graph I build a feature vector containing the mean, median and standard deviation of the node embeddings.
In analysis directory using T-SNE is possible to see the feature vectors in 2D space. 

