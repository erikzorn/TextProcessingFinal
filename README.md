# TextProcessingFinal

## Background
We implemented k-means clustering on a dataset of fake news articles to classify documents based on their topics. This involved creating and term-frequency matrix to track term weights. Once this was achieved, PCA analysis is used to reduce the dimensionality. Next a within sums squares scree plot is shown to the user to enable them to select the optimal cluster amount. Finally, a plot is shown with the articles clustered and their top 3 highest term-frequencies shown.

## Getting Started
clone the repo to your local machine:
```
git clone https://github.com/erikzorn/TextProcessingFinal.git
```
to run the program with python run the following command:
```
python processing.py
```
## Data Source
The fake news data set can be found at https://www.kaggle.com/mrisdal/fake-news.
