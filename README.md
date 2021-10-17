# Graph of words experiments

This project contains the  implementation of my tests with graph of words.

# Datasets
- Polarity
- WEBKB
- 20 Newsgroups
- R8

# Experiments

## First experiment
In my first experiment I created graphs of words with sliding window size is 4, 7, 12 and 20.
I tested some edge weight functions (I will detail more later).
After build a graph for each review, the node embeddings were extracted using **node2vec**.
The node embeddings are fed into a CNN for sentence classification (Text CNN from Kim).
We perform stratified 10-fold cross validation and report the mean acurracy and mean f1 score.
Student t-test and Wilcoxon test are performed to compare different edge weight functions.
